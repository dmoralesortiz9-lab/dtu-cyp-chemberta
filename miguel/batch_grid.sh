#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_grid
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o outputs/grid_%J.out
#BSUB -e outputs/grid_%J.err
set -uo pipefail

# Closes the checkpoint x pooling grid at three seeds. mtr_cls, mtr_mean and
# mlm_cls already have three; mlm_mean is the last cell still on one seed, and
# without it the two effects cannot be separated cleanly.
#
#   bsub < batch_grid.sh
cd "${LS_SUBCWD:?Submit using bsub from the starter directory}"
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
nvidia-smi --query-gpu=name --format=csv,noheader

COMMON="model=chemberta device=cuda model.checkpoint=DeepChem/ChemBERTa-77M-MLM \
model.pooling=mean model.freeze_encoder=false model.augment=false train.epochs=20"

failed=0
for TASK in regression classification; do
    for SEED in 1 2 3; do
        echo "=== $TASK/mlm_mean_s$SEED  ($(date +%H:%M:%S))"
        python -u finetune.py $COMMON task="$TASK" seed=$SEED \
            "hydra.run.dir=runs/$TASK/chemberta/mlm_mean_s${SEED}" \
            || { echo "FAILED: $TASK/mlm_mean_s$SEED"; failed=$((failed + 1)); }
    done
done
echo "=== done, $failed failed"
