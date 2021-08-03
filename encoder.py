#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool
import nltk

from fairseq.data.encoders.gpt2_bpe import get_encoder


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help="path to encoder.json",
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help="path to vocab.bpe",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--outputs-pos",
        nargs="+",
        default=["-"],
        help="path to save pos tagging",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]
        outputs_pos = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs_pos
        ]

        encoder = MultiprocessingEncoder(args)
        # encoder.initializer()
        # for i, line in enumerate(inputs[0]):
        #     line = line.strip()
        #     if line == '':
        #         continue
        #     states, enc_lines, pos_tags, corresponding_words = encoder.encode_lines([line])
        #     # for index, token in enumerate(enc_lines[0].split(' ')):
        #     #     print(encoder.decode([int(token)]), '|', pos_tags[0][index])
        # quit()
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 2)

        stats = Counter()
        for i, (filt, enc_lines, pos_tags, corresponding_words) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
                for pos_tag, output_h in zip(pos_tags, outputs_pos):
                    print(pos_tag, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        pos_tags = []
        corresponding_words = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            dec_line = self.decode(map(int, tokens))
            pos_tag, corresponding_word = self.tag_pos(dec_line, tokens)
            enc_lines.append(" ".join(tokens))
            pos_tags.append(" ".join(pos_tag))
            corresponding_words.append(corresponding_word)
        return ["PASS", enc_lines, pos_tags, corresponding_words]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]
    
    def tag_pos(self, line, enc_tokens):
        tokenizerd_sent = nltk.word_tokenize(line)
        current_token = ''
        cnt = 0
        positions = [0]
        for pos, t in enumerate(line):
            current_token += t
            if tokenizerd_sent[cnt] in current_token or '"' in current_token or "''" in current_token:
                cnt += 1
                positions.append(pos + 1)
                current_token = ''
        try:
            assert len(positions) == len(tokenizerd_sent) + 1,\
            f"positions length {len(positions)} doesn't match tokenized sentence {len(tokenizerd_sent)}"
        except AssertionError as e:
            print(line)
            print(tokenizerd_sent)
            print(positions)
            print(len(positions), len(tokenizerd_sent))
            print()
            quit() 
        poses = nltk.pos_tag(tokenizerd_sent, tagset='universal')
        pos_tag = []
        corresponding_word = []
        cnt = 0
        recover_line = ''
        for index, token in enumerate(enc_tokens):
            recover_line = self.decode(map(int, enc_tokens[:index + 1]))
            if cnt < (len(positions) - 1) and len(recover_line) > positions[cnt + 1]:
                # print("cnt--", cnt)
                # print(t, cnt < (len(positions) - 1), len(recover_line) > positions[cnt + 1])
                cnt += 1
            try:
                pos_tag.append(poses[cnt][1])
            except IndexError as e:
                print("cnt", cnt)
                print(poses)
                print(positions)
                print(len(poses))
                print(len(positions))
                print(positions[-1], len(line))
                print(len(recover_line))
                print(recover_line)
                print(line)
                raise IndexError()
            corresponding_word.append(tokenizerd_sent[cnt])
        return pos_tag, corresponding_word

if __name__ == "__main__":
    main()
