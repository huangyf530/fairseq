#!/bin/bash
#SBATCH --nodelist=101server --gpus=rtx2080:2 --job-name=roberta
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0007          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8        # Number of sequences per batch (batch size)
UPDATE_FREQ=64          # Increase the batch size 16x
export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

ROOT_DIR=.
cd $ROOT_DIR
DATA_DIR=$ROOT_DIR/data-bin/wikitext-103
SAVE_DIR=$ROOT_DIR/checkpoints/roberta
TENSORBOARD_DIR=$SAVE_DIR/tensorboard
LOG_FILE=$SAVE_DIR/train.log
LOG_ARGS="--log-file $LOG_FILE --wandb-project roberta --tensorboard-logdir $TENSORBOARD_DIR"
POS_ARGS="--add-pos --pos-expert-map scripts-mine/pos-expert-map.json"

echo "pip install dependencies..."
# pip install --editable ./
# pip install numpy==1.20.0
# pip install tensorboard

echo "begin training..."
mkdir -p $SAVE_DIR
python train.py --fp16 --fp16-init-scale 8 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES \
    --log-format simple --log-interval 1 $LOG_ARGS \
    --save-dir $SAVE_DIR --save-interval-updates 1000 --keep-interval-updates 3  --keep-last-epochs 5 \
    --base-layers 0 --base-sublayers 1 \
    --validate-interval-updates 500 --skip-invalid-size-inputs-valid-test \
    $POS_ARGS
