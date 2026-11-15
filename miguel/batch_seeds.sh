#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_seeds
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o outputs/seeds_%J.out
#BSUB -e outputs/seeds_%J.err
set -uo pipefail

# Three seeds for the two configurations that still rest on a single run, so the
# two contrasts that matter can carry a p-value like the rest:
#
#   mlm_cls  -> MTR vs MLM   (the main finding; currently one seed per side)
#   mtr_mean -> cls vs mean  (pooling)
#
# Both are compared against mtr_cls_noaug_s{1,2,3}, which already exists. Seeds
# 1-3 match those runs; note this is the TRAINING seed, not the split seed, which
# stays at 42 in data/ and is unaffected (verified: every run shares one
# split_sha256).
#
# The pretraining ablation (rand_*) is deliberately left at one seed: 0.982 vs
# 0.741 and an MCC of exactly 0.000 need no test.
#
#   bsub < batch_seeds.sh
cd "${LS_SUBCWD:?Submit using bsub from the starter directory}"
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
nvidia-smi --query-gpu=name --format=csv,noheader

TASKS="${TASKS:-regression classification}"
MLM=DeepChem/ChemBERTa-77M-MLM
MTR=DeepChem/ChemBERTa-77M-MTR
COMMON="model=chemberta device=cuda model.freeze_encoder=false model.augment=false train.epochs=20"

# name|overrides
RUNS=()
for SEED in 1 2 3; do
    RUNS+=("mlm_cls_s${SEED}|seed=$SEED model.checkpoint=$MLM model.pooling=cls")
    RUNS+=("mtr_mean_s${SEED}|seed=$SEED model.checkpoint=$MTR model.pooling=mean")
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
