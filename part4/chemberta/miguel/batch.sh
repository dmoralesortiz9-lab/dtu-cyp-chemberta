#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_batch
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o outputs/batch_%J.out
#BSUB -e outputs/batch_%J.err
set -uo pipefail

# Several fine-tuning runs inside ONE job: the queue is the bottleneck, not the
# compute, so we wait for a GPU once instead of once per run.
# Edit RUNS below, then:  bsub < batch.sh
cd "${LS_SUBCWD:?Submit using bsub from the starter directory}"
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
nvidia-smi --query-gpu=name --format=csv,noheader

# Passed at submission, e.g.:
#   bsub -env "FREEZE=true" < batch.sh
#   bsub -env "EPOCHS=60, TASKS=classification" < batch.sh
FREEZE="${FREEZE:-false}"
EPOCHS="${EPOCHS:-20}"
TASKS="${TASKS:-regression classification}"
COMMON="model=chemberta device=cuda model.checkpoint=DeepChem/ChemBERTa-77M-MTR model.pooling=cls model.freeze_encoder=$FREEZE train.epochs=$EPOCHS"
prefix=$([ "$FREEZE" = true ] && echo froz_ || echo "")
suffix=$([ "$EPOCHS" = 20 ] && echo "" || echo "_e${EPOCHS}")

# name|extra overrides
RUNS=()
for SEED in 1 2 3; do
    for AUG in false true; do
        tag=$([ "$AUG" = true ] && echo aug || echo noaug)
        RUNS+=("${prefix}mtr_cls_${tag}${suffix}_s${SEED}|seed=$SEED model.augment=$AUG")
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
