import json
import os

from defense.anchor_defense import AnchorDefense
from defense.anchor_trainer import AnchorPretrainer


def get_default_defense_scheme(defense_name):
    if defense_name == "anchor":
        return "AVGuard"
    return "none"


def normalize_defense_args(args):
    if not getattr(args, "defense_scheme", ""):
        args.defense_scheme = get_default_defense_scheme(args.defense)


def prepare_defense(
    args,
    device,
    logger,
    model_list,
    checkpoint,
    clean_train_loader,
    test_loader,
    trigger_dimensions,
):
    if args.defense != "anchor":
        return None

    if args.mode == "pretrain_anchor" and args.force_stage1_retrain:
        pretrainer = AnchorPretrainer(device=device, args=args, logger=logger)
        anchor_defense = pretrainer.pretrain(
            model_list=model_list,
            train_loader=clean_train_loader,
            test_loader=test_loader,
            trigger_dimensions=trigger_dimensions,
        )
        if args.anchor_bank_path:
            anchor_defense.save(args.anchor_bank_path)
            logger.info("=> Saved anchor artifact to '%s'", args.anchor_bank_path)
        return anchor_defense

    if checkpoint and checkpoint.get("anchor_state"):
        logger.info("=> Loading anchor defense state from checkpoint")
        return AnchorDefense.load_from_checkpoint_state(checkpoint["anchor_state"], model_list, device, args, logger)

    stage1_anchor_defense = AnchorDefense.load_stage1_artifacts(model_list, device, args, logger)
    if stage1_anchor_defense is not None:
        return stage1_anchor_defense

    if args.anchor_bank_path and os.path.isfile(args.anchor_bank_path):
        logger.info("=> Loading anchor artifact from '%s'", args.anchor_bank_path)
        return AnchorDefense.load_from_artifact(args.anchor_bank_path, model_list, device, args, logger)

    if args.mode == "test":
        logger.info("=> Anchor defense is enabled but no anchor artifact was found")
        return None

    pretrainer = AnchorPretrainer(device=device, args=args, logger=logger)
    anchor_defense = pretrainer.pretrain(
        model_list=model_list,
        train_loader=clean_train_loader,
        test_loader=test_loader,
        trigger_dimensions=trigger_dimensions,
    )
    if args.anchor_bank_path:
        anchor_defense.save(args.anchor_bank_path)
        logger.info("=> Saved anchor artifact to '%s'", args.anchor_bank_path)
    return anchor_defense


def load_defense_runtime_stats(args, logger, defense_runtime):
    if args.defense != "anchor" or defense_runtime is None:
        return

    stats_path = os.path.join(args.results_dir, "stage2_final_epoch_client_anchor_losses.json")
    if not os.path.isfile(stats_path):
        if defense_runtime.has_stage3_stats():
            logger.info("=> Stage 3 will reuse client anchor losses stored in the loaded anchor state")
        else:
            logger.info("=> Stage 3 client anchor losses were not found at '%s'", stats_path)
        return

    with open(stats_path, "r", encoding="utf-8") as stats_file:
        payload = json.load(stats_file)

    client_anchor_losses = {
        int(client_id): float(client_loss)
        for client_id, client_loss in payload.get("client_anchor_losses", {}).items()
    }
    defense_runtime.set_final_epoch_client_anchor_losses(client_anchor_losses)
    logger.info(
        "=> Loaded Stage 3 client anchor losses from '%s' (epoch %s): %s",
        stats_path,
        payload.get("epoch", "n/a"),
        {client_id: round(client_loss, 6) for client_id, client_loss in sorted(client_anchor_losses.items())},
    )
