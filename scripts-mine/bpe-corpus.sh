# /bin/bash!
if [ ! -d gpt2_bpe ];then
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
fi
for SPLIT in train valid test; do \
    python encoder.py \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs  /home/huangyufei/disk3/wikitext-103/wiki.${SPLIT}.tokens \
        --outputs /home/huangyufei/disk3/wikitext-103/wiki.${SPLIT}.bpe \
        --outputs-pos /home/huangyufei/disk3/wikitext-103/wiki.${SPLIT}.pos \
        --keep-empty \
        --workers 50; \
done