#! /bin/bash
python fairseq_cli/preprocess.py \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref /home/huangyufei/disk3/OpenWebText/openwebtext-raw/openwebtext.train.bpe \
    --validpref /home/huangyufei/disk3/OpenWebText/openwebtext-raw/openwebtext.valid.bpe \
    --testpref /home/huangyufei/disk3/OpenWebText/openwebtext-raw/openwebtext.test.bpe \
    --destdir data-bin/openwebtext \
    --workers 40