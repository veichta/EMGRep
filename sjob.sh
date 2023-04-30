USR="veichta"
DATA="/cluster/scratch/${USR}/nina_db"
LR_CPC=2e-4

for SPLIT in 1 2 3 4 5; do
    TEST_IDX=${SPLIT}
    VAL_IDX=$((SPLIT % 5 + 1))
    echo "python main.py --data_path ${DATA}/data/01_raw --device cuda --debug --wandb --log_dir ${DATA}/logs --lr_cpc ${LR_CPC} --test_idx ${TEST_IDX} --val_idx ${VAL_IDX}"
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
        --mail-user= ${USR}@ethz.ch \
        --wrap="python main.py --data_path ${DATA}/data/01_raw --device cuda --debug --wandb --log_dir ${DATA}/logs --lr_cpc ${LR_CPC} --test_idx ${TEST_IDX} --val_idx ${VAL_IDX}"
done
