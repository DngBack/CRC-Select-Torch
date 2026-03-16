#!/bin/bash
#
# ============================================================
#  CRC-Select — CIFAR-100 Full Pipeline (một lệnh tất cả)
# ============================================================
#
#  Phase A : Train Vanilla SelectiveNet  × N seeds
#  Phase B : Train Deep Gambler          × N seeds
#  Phase C : Train CRC-Select            × N seeds
#  Phase D : Evaluate all baselines      × N seeds
#            (PostHoc-CRC, MSP, Energy, TempScaled, DeepGambler,
#             Vanilla SelectiveNet, CRC-Select)
#
#  Usage:
#    ./run_cifar100_full_pipeline.sh                   # tất cả phases, tất cả seeds
#    ./run_cifar100_full_pipeline.sh --phase A         # chỉ train vanilla
#    ./run_cifar100_full_pipeline.sh --phase B         # chỉ DeepGambler
#    ./run_cifar100_full_pipeline.sh --phase C         # chỉ CRC-Select
#    ./run_cifar100_full_pipeline.sh --phase D         # chỉ evaluate
#    ./run_cifar100_full_pipeline.sh --seeds "42 123"  # subset seeds
#    CUDA_VISIBLE_DEVICES=1 ./run_cifar100_full_pipeline.sh
#
#  Background (recommended):
#    nohup ./run_cifar100_full_pipeline.sh \
#          > logs/training/cifar100/full_pipeline.log 2>&1 &

set -e

GPU=${CUDA_VISIBLE_DEVICES:-0}
PHASE="all"
SEEDS=(42 123 456 789 999)

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)  PHASE="$2";                            shift 2 ;;
        --seeds)  IFS=' ' read -ra SEEDS <<< "$2";       shift 2 ;;
        *) shift ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

PYTHON=/home/duong.xuan.bach/anaconda3/envs/crc-select/bin/python

# ── checkpoint dirs ──────────────────────────────────────────
VANILLA_DIR=checkpoints/vanilla-cifar100
GAMBLER_DIR=checkpoints/DeepGambler-cifar100
CRC_DIR=checkpoints/CRC-Select-cifar100
RESULTS_DIR=results_paper/cifar100

mkdir -p ${VANILLA_DIR} ${GAMBLER_DIR} ${CRC_DIR}
mkdir -p ${RESULTS_DIR}/{posthoc_crc,MSP,Energy,TempScaled_MSP,DeepGambler,CRC-Select,vanilla}
mkdir -p logs/training/cifar100 logs/evaluation/cifar100

# ── shared hyperparams ───────────────────────────────────────
BACKBONE=wrn28_10
DIM_FEATURES=640
DATASET=cifar100
DATAROOT=./data
AUGMENTATION=randcrop
NUM_EPOCHS=300
LR=0.1 ; WD=0.0005 ; MOMENTUM=0.9
SCHED_STEP=60 ; SCHED_GAMMA=0.2
BATCH=128 ; WORKERS=8

ALPHA_RISK=0.1 ; COVERAGE=0.8 ; LM=32.0 ; ALPHA_MIX=0.5
TAU=0.5 ; WARMUP=30 ; RECAL=5 ; MU_INIT=1.0 ; DUAL_LR=0.01 ; DELTA=0.1

COMMON_EVAL="--dataset ${DATASET} --dataroot ${DATAROOT} \
             --backbone ${BACKBONE} --dim_features ${DIM_FEATURES} \
             --batch_size ${BATCH} --num_workers ${WORKERS}"
ALPHA_ARGS="--alpha_values 0.01 0.02 0.05 0.1 0.15 0.2"

errors=()

# ============================================================
#  PHASE A — Vanilla SelectiveNet
# ============================================================
phase_A() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE A: Vanilla SelectiveNet       ║"
    echo "╚══════════════════════════════════════╝"

    for seed in "${SEEDS[@]}"; do
        CKPT="${VANILLA_DIR}/seed_${seed}.pth"
        if [[ -f "$CKPT" ]]; then
            log "SKIP vanilla seed=${seed} (already exists)"
            continue
        fi
        log "START vanilla seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/train.py \
            --seed          ${seed} \
            --backbone      ${BACKBONE} \
            --dim_features  ${DIM_FEATURES} \
            --dataset       ${DATASET} \
            --dataroot      ${DATAROOT} \
            --augmentation  ${AUGMENTATION} \
            --num_epochs    ${NUM_EPOCHS} \
            --lr ${LR} --wd ${WD} --momentum ${MOMENTUM} --nesterov \
            --batch_size    ${BATCH} \
            --num_workers   ${WORKERS} \
            --coverage      ${COVERAGE} \
            --alpha         ${ALPHA_MIX} \
            --checkpoint_dir ${VANILLA_DIR} \
            --unobserve \
            2>&1 | tee "logs/training/cifar100/vanilla_seed_${seed}.log"

        if [[ -f "$CKPT" ]]; then
            log "DONE vanilla seed=${seed}"
        else
            log "ERROR: checkpoint not found after training — ${CKPT}"
            errors+=("vanilla_seed_${seed}")
        fi
    done
    echo "=== Phase A complete ==="
}

# ============================================================
#  PHASE B — Deep Gambler
# ============================================================
phase_B() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE B: Deep Gambler               ║"
    echo "╚══════════════════════════════════════╝"

    for seed in "${SEEDS[@]}"; do
        CKPT="${GAMBLER_DIR}/seed_${seed}.pth"
        if [[ -f "$CKPT" ]]; then
            log "SKIP DeepGambler seed=${seed} (already exists)"
            continue
        fi
        log "START DeepGambler seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/baseline_deep_gambler.py \
            --train \
            --seed          ${seed} \
            --backbone      ${BACKBONE} \
            --dim_features  ${DIM_FEATURES} \
            --dataset       ${DATASET} \
            --dataroot      ${DATAROOT} \
            --augmentation  ${AUGMENTATION} \
            --num_epochs    ${NUM_EPOCHS} \
            --reward        2.2 \
            --scheduler_step  ${SCHED_STEP} \
            --scheduler_gamma ${SCHED_GAMMA} \
            --batch_size    ${BATCH} \
            --num_workers   ${WORKERS} \
            --checkpoint_dir ${GAMBLER_DIR} \
            2>&1 | tee "logs/training/cifar100/gambler_seed_${seed}.log"

        if [[ -f "$CKPT" ]]; then
            log "DONE DeepGambler seed=${seed}"
        else
            log "ERROR: checkpoint not found — ${CKPT}"
            errors+=("gambler_seed_${seed}")
        fi
    done
    echo "=== Phase B complete ==="
}

# ============================================================
#  PHASE C — CRC-Select
# ============================================================
phase_C() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE C: CRC-Select                 ║"
    echo "╚══════════════════════════════════════╝"

    for seed in "${SEEDS[@]}"; do
        CKPT="${CRC_DIR}/seed_${seed}.pth"
        if [[ -f "$CKPT" ]]; then
            log "SKIP CRC-Select seed=${seed} (already exists)"
            continue
        fi
        log "START CRC-Select seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/train_crc_select.py \
            --seed          ${seed} \
            --backbone      ${BACKBONE} \
            --dim_features  ${DIM_FEATURES} \
            --dataset       ${DATASET} \
            --dataroot      ${DATAROOT} \
            --augmentation  ${AUGMENTATION} \
            --num_epochs    ${NUM_EPOCHS} \
            --lr ${LR} --wd ${WD} --momentum ${MOMENTUM} --nesterov \
            --scheduler_step  ${SCHED_STEP} \
            --scheduler_gamma ${SCHED_GAMMA} \
            --batch_size    ${BATCH} \
            --num_workers   ${WORKERS} \
            --alpha_risk    ${ALPHA_RISK} \
            --coverage      ${COVERAGE} \
            --lm            ${LM} \
            --alpha         ${ALPHA_MIX} \
            --tau           ${TAU} \
            --warmup_epochs ${WARMUP} \
            --recalibrate_every ${RECAL} \
            --mu_init       ${MU_INIT} \
            --dual_lr       ${DUAL_LR} \
            --delta         ${DELTA} \
            --use_dual_update \
            --unobserve \
            2>&1 | tee "logs/training/cifar100/crc_select_seed_${seed}.log"

        # train_crc_select.py saves to checkpoints/CRC-Select/seed_N.pth (shared dir)
        # copy to our cifar100-specific dir
        SHARED="checkpoints/CRC-Select/seed_${seed}.pth"
        if [[ -f "$SHARED" && ! -f "$CKPT" ]]; then
            cp "$SHARED" "$CKPT"
        fi

        if [[ -f "$CKPT" ]]; then
            log "DONE CRC-Select seed=${seed}"
        else
            log "ERROR: checkpoint not found — ${CKPT}"
            errors+=("crc_select_seed_${seed}")
        fi
    done
    echo "=== Phase C complete ==="
}

# ============================================================
#  PHASE D — Evaluate all baselines
# ============================================================
phase_D() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE D: Evaluate all baselines     ║"
    echo "╚══════════════════════════════════════╝"

    for seed in "${SEEDS[@]}"; do
        VANILLA_CKPT="${VANILLA_DIR}/seed_${seed}.pth"
        if [[ ! -f "$VANILLA_CKPT" ]]; then
            log "SKIP eval seed=${seed} — vanilla checkpoint missing (run Phase A first)"
            errors+=("eval_no_vanilla_seed_${seed}")
            continue
        fi

        echo ""
        echo "--- seed ${seed} ---"

        # 1. Post-hoc CRC
        DONE="${RESULTS_DIR}/posthoc_crc/seed_${seed}/all_metrics.csv"
        if [[ -f "$DONE" ]]; then
            log "SKIP post-hoc CRC seed=${seed}"
        else
            log "eval: post-hoc CRC seed=${seed}"
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/baseline_posthoc_crc.py \
                -c ${VANILLA_CKPT} --seed ${seed} \
                --alpha_risk ${ALPHA_RISK} --skip_ood \
                ${COMMON_EVAL} -o ${RESULTS_DIR} \
                2>&1 | tee "logs/evaluation/cifar100/posthoc_crc_seed_${seed}.log"
        fi

        # 2. MSP
        DONE="${RESULTS_DIR}/MSP/seed_${seed}/all_metrics.csv"
        if [[ -f "$DONE" ]]; then
            log "SKIP MSP seed=${seed}"
        else
            log "eval: MSP seed=${seed}"
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/baseline_msp.py \
                -c ${VANILLA_CKPT} --seed ${seed} \
                ${COMMON_EVAL} ${ALPHA_ARGS} -o ${RESULTS_DIR} \
                2>&1 | tee "logs/evaluation/cifar100/msp_seed_${seed}.log"
        fi

        # 3. Energy
        DONE="${RESULTS_DIR}/Energy/seed_${seed}/all_metrics.csv"
        if [[ -f "$DONE" ]]; then
            log "SKIP Energy seed=${seed}"
        else
            log "eval: Energy seed=${seed}"
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/baseline_energy.py \
                -c ${VANILLA_CKPT} --seed ${seed} \
                ${COMMON_EVAL} ${ALPHA_ARGS} -o ${RESULTS_DIR} \
                2>&1 | tee "logs/evaluation/cifar100/energy_seed_${seed}.log"
        fi

        # 4. Temperature-scaled MSP
        DONE="${RESULTS_DIR}/TempScaled_MSP/seed_${seed}/all_metrics.csv"
        if [[ -f "$DONE" ]]; then
            log "SKIP TempScaled_MSP seed=${seed}"
        else
            log "eval: TempScaled_MSP seed=${seed}"
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/baseline_temp_scaled.py \
                -c ${VANILLA_CKPT} --seed ${seed} \
                ${COMMON_EVAL} ${ALPHA_ARGS} -o ${RESULTS_DIR} \
                2>&1 | tee "logs/evaluation/cifar100/temp_scaled_seed_${seed}.log"
        fi

        # 5. Deep Gambler
        GAMBLER_CKPT="${GAMBLER_DIR}/seed_${seed}.pth"
        DONE="${RESULTS_DIR}/DeepGambler/seed_${seed}/all_metrics.csv"
        if [[ -f "$DONE" ]]; then
            log "SKIP DeepGambler seed=${seed}"
        elif [[ ! -f "$GAMBLER_CKPT" ]]; then
            log "SKIP DeepGambler eval seed=${seed} — checkpoint missing (run Phase B)"
            errors+=("eval_no_gambler_seed_${seed}")
        else
            log "eval: DeepGambler seed=${seed}"
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/baseline_deep_gambler.py \
                -c ${GAMBLER_CKPT} --seed ${seed} \
                ${COMMON_EVAL} ${ALPHA_ARGS} -o ${RESULTS_DIR} \
                2>&1 | tee "logs/evaluation/cifar100/deep_gambler_seed_${seed}.log"
        fi

        # 6. Vanilla SelectiveNet eval
        DONE="${RESULTS_DIR}/vanilla/seed_${seed}/all_metrics.csv"
        if [[ -f "$DONE" ]]; then
            log "SKIP vanilla eval seed=${seed}"
        else
            log "eval: vanilla seed=${seed}"
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/evaluate_for_paper.py \
                -c ${VANILLA_CKPT} --method_name vanilla --seed ${seed} \
                ${COMMON_EVAL} --alpha_values 0.01 0.02 0.05 0.1 0.15 0.2 \
                --output_dir ${RESULTS_DIR} --skip_ood \
                2>&1 | tee "logs/evaluation/cifar100/vanilla_seed_${seed}.log"
        fi

        # 7. CRC-Select eval
        CRC_CKPT="${CRC_DIR}/seed_${seed}.pth"
        DONE="${RESULTS_DIR}/CRC-Select/seed_${seed}/all_metrics.csv"
        if [[ -f "$DONE" ]]; then
            log "SKIP CRC-Select eval seed=${seed}"
        elif [[ -f "$CRC_CKPT" ]]; then
            log "eval: CRC-Select seed=${seed}"
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/evaluate_for_paper.py \
                -c ${CRC_CKPT} --method_name CRC-Select --seed ${seed} \
                ${COMMON_EVAL} --alpha_values 0.01 0.02 0.05 0.1 0.15 0.2 \
                --output_dir ${RESULTS_DIR} --skip_ood \
                2>&1 | tee "logs/evaluation/cifar100/crc_select_seed_${seed}.log"
        else
            log "SKIP CRC-Select eval seed=${seed} — checkpoint missing (run Phase C)"
        fi

    done
    echo "=== Phase D complete ==="
}

# ============================================================
#  Main
# ============================================================
echo "============================================================"
echo " CRC-Select — CIFAR-100 Full Pipeline"
echo "============================================================"
echo " GPU             : ${GPU}"
echo " Phase           : ${PHASE}"
echo " Seeds           : ${SEEDS[*]}"
echo " Backbone        : ${BACKBONE} (dim=${DIM_FEATURES})"
echo " Augmentation    : ${AUGMENTATION}"
echo " Epochs          : ${NUM_EPOCHS}"
echo " Vanilla ckpts   : ${VANILLA_DIR}"
echo " Gambler ckpts   : ${GAMBLER_DIR}"
echo " CRC-Select ckpts: ${CRC_DIR}"
echo " Results         : ${RESULTS_DIR}"
echo "============================================================"

[[ "$PHASE" == "all" || "$PHASE" == "A" ]] && phase_A
[[ "$PHASE" == "all" || "$PHASE" == "B" ]] && phase_B
[[ "$PHASE" == "all" || "$PHASE" == "C" ]] && phase_C
[[ "$PHASE" == "all" || "$PHASE" == "D" ]] && phase_D

# ============================================================
#  Summary
# ============================================================
echo ""
echo "============================================================"
echo " Summary — CIFAR-100 Results"
echo "============================================================"
for method in CRC-Select vanilla posthoc_crc MSP Energy TempScaled_MSP DeepGambler; do
    echo -n "  ${method}: "
    count=0
    for seed in "${SEEDS[@]}"; do
        [[ -f "${RESULTS_DIR}/${method}/seed_${seed}/all_metrics.csv" ]] && count=$((count+1))
    done
    echo "${count}/${#SEEDS[@]} seeds"
done

if [[ ${#errors[@]} -gt 0 ]]; then
    echo ""
    echo "  Errors/warnings:"
    for e in "${errors[@]}"; do echo "    - ${e}"; done
fi

echo ""
echo " Logs   : logs/{training,evaluation}/cifar100/"
echo " Results: ${RESULTS_DIR}/"
echo "============================================================"
