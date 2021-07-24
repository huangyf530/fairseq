# /bin/bash!
if [ ! -d gpt2_bpe ];then
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
fi
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs /home/huangyufei/disk3/OpenWebText/openwebtext-raw/openwebtext.${SPLIT}.tokens \
        --outputs /home/huangyufei/disk3/OpenWebText/openwebtext-raw/openwebtext.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done