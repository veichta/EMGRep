DATA="/cluster/scratch/bieriv/nina_db"
LR_CPC=2e-4
AR_DIM=256
AR_LAYERS=2
ENC_DIM=256
PREPROCESSING="rms"
NORM="True"
ALPHA=0
DECAY=1e-3
OPTIMIZER="sgd"
SEQ_LEN=5400
SEQ_STRIDE=300
AR_MODEL="gru"
ENCODER_TYPE="cnn"
CLASSIFIER_TYPE="mlp"
POSITIVE_MODE="none" # "none", "session", "subject", "label"
SPLIT_MODE="day"     # "day", "subject", "session"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy

pip install -q -r requirements.txt
for SPLIT in 1 2 3 4 5; do
    VAL_IDX=${SPLIT}
    TEST_IDX=$((SPLIT % 5 + 1))
    echo "python main.py --data ${DATA}/data/01_raw --device cuda --debug --wandb --log_dir ${DATA}/logs --lr_cpc ${LR_CPC} --test_idx ${TEST_IDX} --val_idx ${VAL_IDX} --positive_mode ${POSITIVE_MODE} --encoder_dim ${ENC_DIM} --ar_dim ${AR_DIM} --ar_layers ${AR_LAYERS} --split_mode ${SPLIT_MODE} --preprocessing ${PREPROCESSING} --normalize ${NORM} --cpc_alpha ${ALPHA} --weight_decay_cpc ${DECAY} --optimizer_cpc ${OPTIMIZER} --seq_len ${SEQ_LEN} --seq_stride ${SEQ_STRIDE}  --encoder_type ${ENCODER_TYPE} --ar_model ${AR_MODEL} --classifier_type ${CLASSIFIER_TYPE} --debug
    sbatch \
        --time=06:00:00 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=20 \
        -J "emgrep-split-${SPLIT}" \
        --mem-per-cpu=10000 \
        --gres=gpumem:14240m \
        --gpus=1 \
        --mail-type=ALL \
        --mail-user="${USER}@ethz.ch" \
        --output="logs/${POSITIVE_MODE}-split-${SPLIT}.txt" \
        --wrap="python main.py --data ${DATA}/data/01_raw --device cuda --debug --wandb --log_dir ${DATA}/logs --lr_cpc ${LR_CPC} --test_idx ${TEST_IDX} --val_idx ${VAL_IDX} --positive_mode ${POSITIVE_MODE} --encoder_dim ${ENC_DIM} --ar_dim ${AR_DIM} --ar_layers ${AR_LAYERS} --split_mode ${SPLIT_MODE} --preprocessing ${PREPROCESSING} --normalize ${NORM} --cpc_alpha ${ALPHA} --weight_decay_cpc ${DECAY} --optimizer_cpc ${OPTIMIZER} --seq_len ${SEQ_LEN} --seq_stride ${SEQ_STRIDE}  --encoder_type ${ENCODER_TYPE} --ar_model ${AR_MODEL} --classifier_type ${CLASSIFIER_TYPE} --debug"
done
