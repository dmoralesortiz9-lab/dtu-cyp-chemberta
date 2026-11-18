#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_lr
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o outputs/lr_%J.out
#BSUB -e outputs/lr_%J.err
set -uo pipefail

# Learning-rate sweep for the MTR checkpoint, in ONE job: the queue is the
# bottleneck, not the compute.
#
# Why: every run so far used encoder_lr 1e-5, inherited from atom_jepa.yaml.
# Daniel's sweep on the MLM checkpoint shows that value undertrains, with a
# U-shaped curve bottoming at 1e-4. The optimum may move with the checkpoint,
# so this re-measures it on MTR, and does it over three seeds at the candidate
# value, because single-seed differences here are inside the noise.
#
#   1e-4 x seeds 1,2,3  -> the comparison against the 1e-5 reference
#                          (runs/*/chemberta/mtr_cls_noaug_s{1,2,3}, 3 seeds)
#   3e-5, 3e-4 x seed 1 -> shape of the curve either side of the candidate
#
#   bsub < batch_lr.sh
cd "${LS_SUBCWD:?Submit using bsub from the starter directory}"
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
nvidia-smi --query-gpu=name --format=csv,noheader

TASKS="${TASKS:-regression classification}"
COMMON="model=chemberta device=cuda model.checkpoint=DeepChem/ChemBERTa-77M-MTR \
model.pooling=cls model.freeze_encoder=false model.augment=false train.epochs=20"

# name|overrides   (tag keeps the lr readable in the folder name)
RUNS=()
for SEED in 1 2 3; do
    RUNS+=("mtr_cls_lr1e-4_s${SEED}|seed=$SEED model.encoder_lr=1e-4")
done
RUNS+=("mtr_cls_lr3e-5_s1|seed=1 model.encoder_lr=3e-5")
RUNS+=("mtr_cls_lr3e-4_s1|seed=1 model.encoder_lr=3e-4")

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
