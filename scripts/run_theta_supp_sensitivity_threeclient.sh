#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "main.py" ]]; then
  echo "Please run this script from the project root directory." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-python}"
EXPERIMENT_ROOT="configs/avguard/threeclient_full_avguard_fixed_lambda_anchor_theta_supp_sensitivity"
THETA_SUPPS=("0.05" "0.10" "0.15" "0.20" "0.25")
DATASET_NAMES=("CIFAR10" "NUSWIDE" "PHISHING" "IEEE_CIS_FRAUD")
CONFIG_DIRS=("cifar10" "nuswide" "phishing" "ieee_cis_fraud")
STAGES=("stage1:stage1.json" "stage2:stage2.json" "stage3:stage3_test.json")

run_stage() {
  local dataset="$1"
  local theta_supp="$2"
  local stage_name="$3"
  local config_path="$4"

  if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path" >&2
    exit 1
  fi

  echo "Running dataset=${dataset} theta_supp=${theta_supp} stage=${stage_name}"
  echo "${PYTHON_BIN} main.py --config \"${config_path}\""

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi

  "$PYTHON_BIN" main.py --config "$config_path"
}

for theta_supp in "${THETA_SUPPS[@]}"; do
  for index in "${!DATASET_NAMES[@]}"; do
    dataset="${DATASET_NAMES[$index]}"
    config_dir="${CONFIG_DIRS[$index]}"
    run_dir="${EXPERIMENT_ROOT}/theta_supp_${theta_supp}/${config_dir}"

    if [[ "${STAGE3_ONLY:-0}" == "1" ]]; then
      run_stage "$dataset" "$theta_supp" "stage3" "${run_dir}/stage3_test.json"
      continue
    fi

    for stage_spec in "${STAGES[@]}"; do
      stage_name="${stage_spec%%:*}"
      config_file="${stage_spec#*:}"
      run_stage "$dataset" "$theta_supp" "$stage_name" "${run_dir}/${config_file}"
    done
  done
done

echo "All three-client theta_supp sensitivity experiments completed."
