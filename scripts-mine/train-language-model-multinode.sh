#!/bin/bash
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$DLS_TASK_NUMBER))
MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
MASTER_ADDR="${MASTER_HOST%%:*}"
MASTER_PORT="${MASTER_HOST##*:}"
NNODES="$DLS_TASK_NUMBER"
NODE_RANK="$DLS_TASK_INDEX"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

echo "Distributed args: ${DISTRIBUTED_ARGS}"

ROOT_DIR=/workspace
cd $ROOT_DIR
DATA_DIR=$ROOT_DIR/data-bin/openwebtext
SAVE_DIR=$ROOT_DIR/checkpoints/language-model-16expert-3layer
TENSORBOARD_DIR=$SAVE_DIR/tensorboard
LOG_FILE=$SAVE_DIR/train.log
LOG_ARGS="--log-file $LOG_FILE --tensorboard-logdir $TENSORBOARD_DIR"
BASE_LAYERS=3

echo "pip install dependencies..."
# pip install --user --editable ./
# pip install numpy==1.20.0
# pip install tensorboard
python setup.py build_ext --inplace

echo "[Train] begin training..."
mkdir -p $SAVE_DIR
python -m torch.distributed.launch ${DISTRIBUTED_ARGS} train.py \
    --fp16 --fp16-init-scale 8 $DATA_DIR \
    --task language_modeling \
    --arch transformer_lm_gpt --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --warmup-init-lr 1e-07 --max-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 \
    --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES \
    --log-format simple --log-interval 10 $LOG_ARGS \
    --save-dir $SAVE_DIR --save-interval-updates 1000 --keep-interval-updates 3  --keep-last-epochs 5 \
    --base-layers $BASE_LAYERS --base-sublayers 1 \
    --validate-interval-updates 500 --skip-invalid-size-inputs-valid-test