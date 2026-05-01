import json
import os

from defense.anchor_defense import AnchorDefense
from defense.anchor_trainer import AnchorPretrainer


def get_default_defense_scheme():
    return "AVGuard"


def normalize_defense_args(args):
    if args.defense == "vflip":
        raise ValueError(
            "The VFLIP defense scheme has been removed from this project. "
            "Please switch to defense='anchor' for the AVGuard pipeline."
        )
    if args.defense != "anchor":
        raise ValueError(
            "Unsupported defense '{}'. This workspace now keeps only the AVGuard pipeline with defense='anchor'.".format(
                args.defense
            )
        )
    if not getattr(args, "defense_scheme", ""):
        args.defense_scheme = get_default_defense_scheme()


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
    stage1_enabled = bool(getattr(args, "enable_stage1", True))

    if args.mode == "pretrain_anchor" and args.force_stage1_retrain and stage1_enabled:
        pretrainer = AnchorPretrainer(device=device, args=args, logger=logger)
        anchor_defense = pretrainer.pretrain(
            model_list=model_list,
            train_loader=clean_train_loader,
            test_loader=test_loader,
            trigger_dimensions=trigger_dimensions,
            checkpoint=checkpoint,
        )
        if args.anchor_bank_path:
            anchor_defense.save(args.anchor_bank_path)
            logger.info("=> Saved anchor artifact to '%s'", args.anchor_bank_path)
        return anchor_defense

    if checkpoint and checkpoint.get("anchor_state"):
        logger.info("=> Loading anchor defense state from checkpoint")
        return AnchorDefense.load_from_checkpoint_state(checkpoint["anchor_state"], model_list, device, args, logger)

    if not stage1_enabled:
        logger.info(
            "=> Stage 1 is disabled by config; skipping Stage 1 artifact reuse/pretraining and creating random normalized anchors"
        )
        anchor_defense = AnchorDefense.create_with_random_anchors(model_list, device, args, logger)
        if args.mode == "pretrain_anchor" and args.anchor_bank_path:
            anchor_defense.save(args.anchor_bank_path)
            logger.info("=> Saved random-initialized anchor artifact to '%s'", args.anchor_bank_path)
        return anchor_defense

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
        checkpoint=checkpoint,
    )
    if args.anchor_bank_path:
        anchor_defense.save(args.anchor_bank_path)
        logger.info("=> Saved anchor artifact to '%s'", args.anchor_bank_path)
    return anchor_defense


def load_defense_runtime_stats(args, logger, defense_runtime):
    if defense_runtime is None:
        return

    stats_path = os.path.join(args.results_dir, "stage2_final_epoch_client_anchor_losses.json")
    if not os.path.isfile(stats_path):
        if defense_runtime.has_stage3_stats():
            logger.info(
                "=> Stage 3 is ready; it will use Stage 1 client reliability from the loaded anchor state when available. Stage 2 client anchor-loss stats are optional diagnostics."
            )
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
        "=> Loaded optional Stage 2 client anchor-loss diagnostics from '%s' (epoch %s): %s",
        stats_path,
        payload.get("epoch", "n/a"),
        {client_id: round(client_loss, 6) for client_id, client_loss in sorted(client_anchor_losses.items())},
    )
