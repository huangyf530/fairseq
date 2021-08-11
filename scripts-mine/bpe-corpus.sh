# /bin/bash!
if [ ! -d gpt2_bpe ];then
    mkdir -p gpt2_bpe
    wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
fi
DIR="/home/huangyufei/disk3/OpenWebText/openwebtext-raw"
for SPLIT in train valid test; do \
    python encoder.py \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs  ${DIR}/openwebtext.${SPLIT}.tokens \
        --outputs ${DIR}/openwebtext.${SPLIT}.bpe \
        --outputs-pos ${DIR}/openwebtext.${SPLIT}.pos \
        --keep-empty \
        --workers 50; \
done
# python encoder.py \
#         --encoder-json gpt2_bpe/encoder.json \
#         --vocab-bpe gpt2_bpe/vocab.bpe \
#         --inputs  ${DIR}/test.tokens \
#         --outputs ${DIR}/test.bpe \
#         --outputs-pos ${DIR}/test.pos \
#         --keep-empty \
#         --workers 1; \