#!/bin/bash
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)

ROOT_DIR=/workspace
cd $ROOT_DIR
DATA_DIR=$ROOT_DIR/data-bin/wikitext-103
MODEL_DIR=$ROOT_DIR/checkpoints/language-model
MODEL_PATH=$MODEL_DIR/checkpoint_best.pt

# echo "pip install dependencies..."
# pip install --user --editable ./
# pip install numpy==1.20.0
# pip install tensorboard
# python setup.py build_ext --inplace
mkdir -p $MODEL_DIR
echo "[Eval] begin elvaluate..."
python validate.py $DATA_DIR \
    --task language_modeling \
    --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
    --distributed-world-size 2 \
    --batch-size $MAX_SENTENCES \
    --log-format simple --log-interval 1 \
    --path $MODEL_PATH \
    --skip-invalid-size-inputs-valid-test
