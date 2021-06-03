# /bin/bash!
# mkdir -p gpt2_bpe
# wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
# wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs ../bert-corpus/bert-corpus.${SPLIT}.raw \
        --outputs ../bert-corpus/bert-corpus.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done