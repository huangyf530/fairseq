#! /bin/bash
python fairseq_cli/preprocess.py \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref ../bert-corpus/bert-corpus.train.bpe \
    --validpref ../bert-corpus/bert-corpus.valid.bpe \
    --testpref ../bert-corpus/bert-corpus.test.bpe \
    --destdir data-bin/bert-corpus \
    --workers 40