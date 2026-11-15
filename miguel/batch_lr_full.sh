#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_lrfull
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o outputs/lrfull_%J.out
#BSUB -e outputs/lrfull_%J.err
set -uo pipefail

# Completes the learning-rate curve for MTR + CLS, three seeds at every point.
#
# Why: the sweep so far covers 1e-5 to 3e-4 and finds a flat plateau, but two of
# those points have a single seed, and a reader can always say the range was too
# narrow to see anything. Going out to 1e-6 and 1e-3 should break the model on
# both sides, which is what shows the plateau is a real result and not a lack of
# resolution: "we swept three orders of magnitude; it is insensitive between
# 3e-5 and 3e-4 and degrades outside".
#
# Already available with three seeds: 1e-5 (mtr_cls_noaug_s*), 1e-4.
# Added here: 1e-6 and 1e-3 (new), and 3e-5 and 3e-4 brought up to three seeds.
#
# Note on multiple comparisons: this makes five contrasts against the reference,
# so the honest threshold is Bonferroni 0.05/5 = 0.01, not 0.05.
#
#   bsub < batch_lr_full.sh
cd "${LS_SUBCWD:?Submit using bsub from the starter directory}"
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
nvidia-smi --query-gpu=name --format=csv,noheader

TASKS="${TASKS:-regression classification}"
COMMON="model=chemberta device=cuda model.checkpoint=DeepChem/ChemBERTa-77M-MTR \
model.pooling=cls model.freeze_encoder=false model.augment=false train.epochs=20"

RUNS=()
for SEED in 1 2 3; do
    for LR in 1e-6 3e-5 3e-4 1e-3; do
        # 3e-5 and 3e-4 already have s1; skip it instead of overwriting.
        if [[ ( "$LR" == "3e-5" || "$LR" == "3e-4" ) && "$SEED" == "1" ]]; then continue; fi
        RUNS+=("mtr_cls_lr${LR}_s${SEED}|seed=$SEED model.encoder_lr=$LR")
    done
done

failed=0
for TASK in $TASKS; do
    for SPEC in "${RUNS[@]}"; do
        NAME="${SPEC%%|*}"; OV="${SPEC#*|}"
        echo "=== $TASK/$NAME  ($(date +%H:%M:%S))"
        python -u finetune.py $COMMON task="$TASK" $OV \
            "hydra.run.dir=runs/$TASK/chemberta/$NAME" \
            || { echo "FAILED: $TASK/$NAME"; failed=$((failed + 1)); }
    done
done
echo "=== done, $failed failed"
