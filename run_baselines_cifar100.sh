#!/bin/bash
#
# Baseline Evaluations for CIFAR-100
#
# Prerequisites:
#   - checkpoints/vanilla-cifar100/seed_*.pth       (run train_cifar100_vanilla.sh Phase A)
#   - checkpoints/DeepGambler-cifar100/seed_*.pth   (run train_cifar100_vanilla.sh Phase B)
#   - checkpoints/CRC-Select-cifar100/seed_*.pth    (run train_cifar100_multi_seed.sh)
#
# Usage:
#   ./run_baselines_cifar100.sh
#   ./run_baselines_cifar100.sh --seeds "42 123"
#   ./run_baselines_cifar100.sh --skip-gambler

set -e

GPU=${CUDA_VISIBLE_DEVICES:-0}
SEEDS=(42 123 456 789 999)
SKIP_GAMBLER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)        IFS=' ' read -ra SEEDS <<< "$2"; shift 2 ;;
        --skip-gambler) SKIP_GAMBLER=true; shift ;;
        *) shift ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p logs/evaluation/cifar100
mkdir -p results_paper/cifar100/{posthoc_crc,MSP,Energy,TempScaled_MSP,DeepGambler,CRC-Select,vanilla}

echo "============================================================"
echo " CRC-Select: CIFAR-100 Baseline Evaluations"
echo "============================================================"
echo " GPU: ${GPU}"
echo " Seeds: ${SEEDS[*]}"
echo " Skip DeepGambler: ${SKIP_GAMBLER}"
echo "============================================================"

# Backbone used for ALL CIFAR-100 checkpoints
BACKBONE=wrn28_10
DIM_FEATURES=640

COMMON_ARGS="--dataset cifar100 --dataroot ./data --batch_size 128 --num_workers 4 \
             --backbone ${BACKBONE} --dim_features ${DIM_FEATURES}"
ALPHA_ARGS="--alpha_values 0.01 0.02 0.05 0.1 0.15 0.2"

errors=()

for seed in "${SEEDS[@]}"; do
    VANILLA_CKPT="checkpoints/vanilla-cifar100/seed_${seed}.pth"

    if [[ ! -f "$VANILLA_CKPT" ]]; then
        log "WARNING: Missing vanilla checkpoint for seed=${seed}: ${VANILLA_CKPT}"
        log "  Run ./train_cifar100_vanilla.sh first"
        errors+=("missing_vanilla_cifar100_seed_${seed}")
        continue
    fi

    echo ""
    echo "=== Seed ${seed} ==="

    # ----------------------------------------------------------
    # 1. Post-hoc CRC
    # ----------------------------------------------------------
    POSTHOC_DONE="results_paper/cifar100/posthoc_crc/seed_${seed}/all_metrics.csv"
    if [[ -f "$POSTHOC_DONE" ]]; then
        log "SKIP post-hoc CRC seed=${seed} (results exist)"
    else
        log "START post-hoc CRC seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_posthoc_crc.py \
            -c ${VANILLA_CKPT} \
            --seed ${seed} \
            --alpha_risk 0.1 \
            --skip_ood \
            ${COMMON_ARGS} \
            -o ./results_paper/cifar100 \
            2>&1 | tee logs/evaluation/cifar100/posthoc_crc_seed_${seed}.log
        log "DONE post-hoc CRC seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 2. MSP
    # ----------------------------------------------------------
    MSP_DONE="results_paper/cifar100/MSP/seed_${seed}/all_metrics.csv"
    if [[ -f "$MSP_DONE" ]]; then
        log "SKIP MSP seed=${seed} (results exist)"
    else
        log "START MSP seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_msp.py \
            -c ${VANILLA_CKPT} \
            --seed ${seed} \
            ${COMMON_ARGS} \
            ${ALPHA_ARGS} \
            -o ./results_paper/cifar100 \
            2>&1 | tee logs/evaluation/cifar100/msp_seed_${seed}.log
        log "DONE MSP seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 3. Energy
    # ----------------------------------------------------------
    ENERGY_DONE="results_paper/cifar100/Energy/seed_${seed}/all_metrics.csv"
    if [[ -f "$ENERGY_DONE" ]]; then
        log "SKIP Energy seed=${seed} (results exist)"
    else
        log "START Energy seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_energy.py \
            -c ${VANILLA_CKPT} \
            --seed ${seed} \
            ${COMMON_ARGS} \
            ${ALPHA_ARGS} \
            -o ./results_paper/cifar100 \
            2>&1 | tee logs/evaluation/cifar100/energy_seed_${seed}.log
        log "DONE Energy seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 4. Temperature-Scaled MSP
    # ----------------------------------------------------------
    TEMP_DONE="results_paper/cifar100/TempScaled_MSP/seed_${seed}/all_metrics.csv"
    if [[ -f "$TEMP_DONE" ]]; then
        log "SKIP TempScaled_MSP seed=${seed} (results exist)"
    else
        log "START TempScaled_MSP seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_temp_scaled.py \
            -c ${VANILLA_CKPT} \
            --seed ${seed} \
            ${COMMON_ARGS} \
            ${ALPHA_ARGS} \
            -o ./results_paper/cifar100 \
            2>&1 | tee logs/evaluation/cifar100/temp_scaled_seed_${seed}.log
        log "DONE TempScaled_MSP seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 5. Deep Gambler
    # ----------------------------------------------------------
    if [[ "$SKIP_GAMBLER" == "false" ]]; then
        GAMBLER_CKPT="checkpoints/DeepGambler-cifar100/seed_${seed}.pth"
        GAMBLER_DONE="results_paper/cifar100/DeepGambler/seed_${seed}/all_metrics.csv"
        if [[ -f "$GAMBLER_DONE" ]]; then
            log "SKIP DeepGambler seed=${seed} (results exist)"
        elif [[ ! -f "$GAMBLER_CKPT" ]]; then
            log "SKIP DeepGambler seed=${seed} — checkpoint missing (run train_cifar100_gambler.sh)"
            errors+=("missing_gambler_cifar100_ckpt_seed_${seed}")
        else
            log "START DeepGambler seed=${seed}"
            CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_deep_gambler.py \
                -c ${GAMBLER_CKPT} \
                --seed ${seed} \
                ${COMMON_ARGS} \
                ${ALPHA_ARGS} \
                -o ./results_paper/cifar100 \
                2>&1 | tee logs/evaluation/cifar100/deep_gambler_seed_${seed}.log
            log "DONE DeepGambler seed=${seed}"
        fi
    fi

    # ----------------------------------------------------------
    # 6. Vanilla SelectiveNet
    # ----------------------------------------------------------
    VANILLA_EVAL_DONE="results_paper/cifar100/vanilla/seed_${seed}/all_metrics.csv"
    if [[ -f "$VANILLA_EVAL_DONE" ]]; then
        log "SKIP Vanilla eval seed=${seed} (results exist)"
    else
        log "START Vanilla eval seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/evaluate_for_paper.py \
            -c ${VANILLA_CKPT} \
            --method_name vanilla \
            --seed ${seed} \
            ${COMMON_ARGS} \
            --alpha_values 0.01 0.02 0.05 0.1 0.15 0.2 \
            --output_dir ./results_paper/cifar100 \
            --skip_ood \
            2>&1 | tee logs/evaluation/cifar100/vanilla_seed_${seed}.log
        log "DONE Vanilla eval seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 7. CRC-Select
    # ----------------------------------------------------------
    CRC_CKPT="checkpoints/CRC-Select-cifar100/seed_${seed}.pth"
    CRC_DONE="results_paper/cifar100/CRC-Select/seed_${seed}/all_metrics.csv"
    if [[ -f "$CRC_DONE" ]]; then
        log "SKIP CRC-Select eval seed=${seed} (results exist)"
    elif [[ -f "$CRC_CKPT" ]]; then
        log "START CRC-Select eval seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/evaluate_for_paper.py \
            -c ${CRC_CKPT} \
            --method_name CRC-Select \
            --seed ${seed} \
            ${COMMON_ARGS} \
            --alpha_values 0.01 0.02 0.05 0.1 0.15 0.2 \
            --output_dir ./results_paper/cifar100 \
            --skip_ood \
            2>&1 | tee logs/evaluation/cifar100/crc_select_seed_${seed}.log
        log "DONE CRC-Select eval seed=${seed}"
    else
        log "INFO: No CRC-Select checkpoint for seed=${seed} (run train_cifar100_multi_seed.sh)"
    fi

done

echo ""
echo "============================================================"
echo " CIFAR-100 Evaluation complete. Results summary:"
echo "============================================================"
for method in CRC-Select vanilla posthoc_crc MSP Energy TempScaled_MSP DeepGambler; do
    echo -n "  ${method}: "
    count=0
    for seed in "${SEEDS[@]}"; do
        [[ -f "results_paper/cifar100/${method}/seed_${seed}/all_metrics.csv" ]] && count=$((count+1))
    done
    echo "${count}/${#SEEDS[@]} seeds"
done

if [[ ${#errors[@]} -gt 0 ]]; then
    echo ""
    echo "  WARNINGS:"
    for e in "${errors[@]}"; do echo "    - $e"; done
fi
