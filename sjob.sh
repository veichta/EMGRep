USR="veichta"
DATA="/cluster/scratch/${USR}/nina_db"
LR_CPC=2e-4
AR_DIM=256
AR_LAYERS=2
ENC_DIM=256

TASK="none" # "none", "session", "subject", "label"

for SPLIT in 1 2 3 4 5; do
    VAL_IDX=${SPLIT}
    TEST_IDX=$((SPLIT % 5 + 1))
    echo "python main.py --data ${DATA}/data/01_raw --device cuda --debug --wandb --log_dir ${DATA}/logs --lr_cpc ${LR_CPC} --test_idx ${TEST_IDX} --val_idx ${VAL_IDX} --positive_mode ${TASK} --encoder_dim ${ENC_DIM} --ar_dim ${AR_DIM} --ar_layers ${AR_LAYERS}"
    sbatch \
        --time=05:00:00 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=20 \
        -J "emgrep-split-${SPLIT}" \
        --mem-per-cpu=5000 \
        --gres=gpumem:10240m \
        --gpus=1 \
        --mail-type=ALL \
        --mail-user="${USR}@ethz.ch" \
        --output="logs/${TASK}-split-${SPLIT}.txt" \
        --wrap="python main.py --data ${DATA}/data/01_raw --device cuda --debug --wandb --log_dir ${DATA}/logs --lr_cpc ${LR_CPC} --test_idx ${TEST_IDX} --val_idx ${VAL_IDX} --positive_mode ${TASK} --encoder_dim ${ENC_DIM} --ar_dim ${AR_DIM} --ar_layers ${AR_LAYERS}"
done
