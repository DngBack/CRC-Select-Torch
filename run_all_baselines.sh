#!/bin/bash
#
# Baseline Evaluation Script — runs all 5 baselines for all seeds
#
# Prerequisites:
#   - checkpoints/vanilla/seed_{42,123,456,999}.pth  (from run_training_sequential.sh Phase A)
#   - checkpoints/DeepGambler/seed_{42,123,456,999}.pth (from Phase B)
#
# Evaluations run (per seed):
#   1. Post-hoc CRC  → results_paper/posthoc_crc/seed_N/
#   2. MSP           → results_paper/MSP/seed_N/
#   3. Energy        → results_paper/Energy/seed_N/
#   4. TempScaled    → results_paper/TempScaled_MSP/seed_N/
#   5. DeepGambler   → results_paper/DeepGambler/seed_N/
#   6. CRC-Select    → results_paper/CRC-Select/seed_N/  (if checkpoint exists from Phase C)
#
# Usage:
#   ./run_all_baselines.sh                     # All seeds and baselines
#   ./run_all_baselines.sh --seeds "42 123"    # Specific seeds
#   ./run_all_baselines.sh --skip-gambler      # Skip DeepGambler (if not yet trained)
#
# Estimated wall time: ~3-4h total (no GPU training, just forward passes)

set -e

GPU=${CUDA_VISIBLE_DEVICES:-0}
SEEDS=(42 123 456 999)
SKIP_GAMBLER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)   IFS=' ' read -ra SEEDS <<< "$2"; shift 2 ;;
        --skip-gambler) SKIP_GAMBLER=true; shift ;;
        *) shift ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p logs/evaluation
mkdir -p results_paper/{posthoc_crc,MSP,Energy,TempScaled_MSP,DeepGambler,CRC-Select}

echo "============================================================"
echo " CRC-Select Paper: Baseline Evaluations"
echo "============================================================"
echo " GPU: ${GPU}"
echo " Seeds: ${SEEDS[*]}"
echo " Skip DeepGambler: ${SKIP_GAMBLER}"
echo "============================================================"

COMMON_ARGS="--dataset cifar10 --dataroot ./data --batch_size 128 --num_workers 4"
ALPHA_ARGS="--alpha_values 0.01 0.02 0.05 0.1 0.15 0.2"

errors=()

for seed in "${SEEDS[@]}"; do
    VANILLA_CKPT="checkpoints/vanilla/seed_${seed}.pth"

    if [[ ! -f "$VANILLA_CKPT" ]]; then
        log "WARNING: Missing vanilla checkpoint for seed=${seed}: ${VANILLA_CKPT}"
        log "  Run ./run_training_sequential.sh --phase A first"
        errors+=("missing_vanilla_seed_${seed}")
        continue
    fi

    echo ""
    echo "=== Seed ${seed} ==="

    # ----------------------------------------------------------
    # 1. Post-hoc CRC
    # ----------------------------------------------------------
    POSTHOC_DONE="results_paper/posthoc_crc/seed_${seed}/all_metrics.csv"
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
            -o ./results_paper \
            2>&1 | tee logs/evaluation/posthoc_crc_seed_${seed}.log
        log "DONE post-hoc CRC seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 2. MSP
    # ----------------------------------------------------------
    MSP_DONE="results_paper/MSP/seed_${seed}/all_metrics.csv"
    if [[ -f "$MSP_DONE" ]]; then
        log "SKIP MSP seed=${seed} (results exist)"
    else
        log "START MSP seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_msp.py \
            -c ${VANILLA_CKPT} \
            --seed ${seed} \
            ${COMMON_ARGS} \
            ${ALPHA_ARGS} \
            -o ./results_paper \
            2>&1 | tee logs/evaluation/msp_seed_${seed}.log
        log "DONE MSP seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 3. Energy
    # ----------------------------------------------------------
    ENERGY_DONE="results_paper/Energy/seed_${seed}/all_metrics.csv"
    if [[ -f "$ENERGY_DONE" ]]; then
        log "SKIP Energy seed=${seed} (results exist)"
    else
        log "START Energy seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_energy.py \
            -c ${VANILLA_CKPT} \
            --seed ${seed} \
            ${COMMON_ARGS} \
            ${ALPHA_ARGS} \
            -o ./results_paper \
            2>&1 | tee logs/evaluation/energy_seed_${seed}.log
        log "DONE Energy seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 4. Temperature-Scaled MSP
    # ----------------------------------------------------------
    TEMP_DONE="results_paper/TempScaled_MSP/seed_${seed}/all_metrics.csv"
    if [[ -f "$TEMP_DONE" ]]; then
        log "SKIP TempScaled_MSP seed=${seed} (results exist)"
    else
        log "START TempScaled_MSP seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_temp_scaled.py \
            -c ${VANILLA_CKPT} \
            --seed ${seed} \
            ${COMMON_ARGS} \
            ${ALPHA_ARGS} \
            -o ./results_paper \
            2>&1 | tee logs/evaluation/temp_scaled_seed_${seed}.log
        log "DONE TempScaled_MSP seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 5. Deep Gambler
    # ----------------------------------------------------------
    if [[ "$SKIP_GAMBLER" == "false" ]]; then
        GAMBLER_CKPT="checkpoints/DeepGambler/seed_${seed}.pth"
        GAMBLER_DONE="results_paper/DeepGambler/seed_${seed}/all_metrics.csv"
        if [[ -f "$GAMBLER_DONE" ]]; then
            log "SKIP DeepGambler seed=${seed} (results exist)"
        elif [[ ! -f "$GAMBLER_CKPT" ]]; then
            log "SKIP DeepGambler seed=${seed} — checkpoint missing: ${GAMBLER_CKPT}"
            errors+=("missing_gambler_ckpt_seed_${seed}")
        else
            log "START DeepGambler seed=${seed}"
            CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_deep_gambler.py \
                -c ${GAMBLER_CKPT} \
                --seed ${seed} \
                ${COMMON_ARGS} \
                ${ALPHA_ARGS} \
                -o ./results_paper \
                2>&1 | tee logs/evaluation/deep_gambler_seed_${seed}.log
            log "DONE DeepGambler seed=${seed}"
        fi
    fi

    # ----------------------------------------------------------
    # 6. Vanilla evaluation (SelectiveNet selector as acceptance score)
    # ----------------------------------------------------------
    VANILLA_EVAL_DONE="results_paper/vanilla/seed_${seed}/all_metrics.csv"
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
            --output_dir ./results_paper \
            --skip_ood \
            2>&1 | tee logs/evaluation/vanilla_seed_${seed}.log
        log "DONE Vanilla eval seed=${seed}"
    fi

    # ----------------------------------------------------------
    # 7. CRC-Select evaluation (if retrained checkpoint exists)
    # ----------------------------------------------------------
    CRC_CKPT="checkpoints/CRC-Select/seed_${seed}.pth"
    CRC_DONE="results_paper/CRC-Select/seed_${seed}/all_metrics.csv"
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
            --output_dir ./results_paper \
            --skip_ood \
            2>&1 | tee logs/evaluation/crc_select_seed_${seed}.log
        log "DONE CRC-Select eval seed=${seed}"
    else
        log "INFO: No CRC-Select checkpoint for seed=${seed} — skipping (using existing CSV results)"
    fi

done

# Final status report
echo ""
echo "============================================================"
echo " Evaluation complete. Results summary:"
echo "============================================================"
for method in CRC-Select vanilla posthoc_crc MSP Energy TempScaled_MSP DeepGambler; do
    echo -n "  ${method}: "
    count=0
    for seed in "${SEEDS[@]}"; do
        if [[ -f "results_paper/${method}/seed_${seed}/all_metrics.csv" ]]; then
            count=$((count+1))
        fi
    done
    echo "${count}/${#SEEDS[@]} seeds"
done

if [[ ${#errors[@]} -gt 0 ]]; then
    echo ""
    echo "  WARNINGS:"
    for e in "${errors[@]}"; do echo "    - $e"; done
fi

echo ""
echo " Next step: ./run_final_analysis.sh"
