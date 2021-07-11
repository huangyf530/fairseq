# !/bin/bash
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0007          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8         # Number of sequences per batch (batch size)
UPDATE_FREQ=64          # Increase the batch size 32x
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO

GPUS_PER_NODE=2
# WORLD_SIZE=$(($GPUS_PER_NODE*$DLS_TASK_NUMBER))
# MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
MASTER_HOST="192.168.1.101:10001"
MASTER_ADDR="${MASTER_HOST%%:*}"
MASTER_PORT="${MASTER_HOST##*:}"
# NNODES="$DLS_TASK_NUMBER"
# NODE_RANK="$DLS_TASK_INDEX"
NNODES=2
NODE_RANK=$1
# WORLD_SIZE=$WORLD_SIZE
# RANK=$DLS_TASK_INDEX
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

DATA_DIR=data-bin/bert-corpus         # 数据文件地址，这里写的是相对地址
SAVE_DIR=checkpoints/roberta-expert-test   # 保存模型和log的位置，这里写的也是相对地址
TENSORBOARD_DIR=$SAVE_DIR/tensorboard
LOG_FILE=$SAVE_DIR/train.log
LOG_ARGS="--log-file $LOG_FILE --tensorboard-logdir $TENSORBOARD_DIR"
# 进入当前文件夹挂载目录
# echo "[Fairseq] Build c++ extensible..."
# python setup.py build_ext --inplace
mkdir -p $SAVE_DIR
echo "[Fairseq] Begin Training ..."
echo "[Fairseq] $DISTRIBUTED_ARGS"
python -m torch.distributed.launch ${DISTRIBUTED_ARGS} fairseq_cli/train.py \
    --fp16 --fp16-init-scale 8 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES \
    --log-format simple --log-interval 1 \
    --save-dir $SAVE_DIR --save-interval-updates 1000 --keep-interval-updates 3 \
    --base-layers 12 --base-sublayers 1 \
    --validate-interval-updates 500 --skip-invalid-size-inputs-valid-test \
    --ddp-backend legacy_ddp --fp16-no-flatten-grads
