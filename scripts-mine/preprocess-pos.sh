#! /bin/bash
ROOT_DIR=/home/huangyufei/disk3/wikitext-103
DEST_DIR=data-bin/wikitext-103-pos
MV_DIR=data-bin/wikitext-103

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref ${ROOT_DIR}/wiki.train.pos \
    --validpref ${ROOT_DIR}/wiki.valid.pos \
    --testpref ${ROOT_DIR}/wiki.test.pos \
    --destdir ${DEST_DIR} \
    --workers 50

for SPLIT in train valid test; do \
    mv ${DEST_DIR}/${SPLIT}.idx ${MV_DIR}/pos-${SPLIT}.idx
    mv ${DEST_DIR}/${SPLIT}.bin ${MV_DIR}/pos-${SPLIT}.bin
done
mv ${DEST_DIR}/dict.txt ${MV_DIR}/pos-dict.txt