# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ofasys.preprocessor.tokenizer.gpt2_bpe import GPT2BPE


class Encoder(object):
    def __init__(self, args):
        self.args = args
        Encoder.tokenizer = GPT2BPE()

    def encode(self, sentence):
        if self.args.json_key:
            sentence = json.loads(sentence)
            sentence = sentence[self.args.json_key]
        tokens = Encoder.tokenizer._encode(" " + sentence)
        if len(tokens) > 0:  # and self.args.append_eod:
            tokens.append(Encoder.tokenizer.eod)
        return tokens, len(sentence)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('input', type=str, help='Path to input text')
    group.add_argument('--json-key', default=None, type=str, help='json key')

    # group = parser.add_argument_group(title='tokenizer')
    # group.add_argument('--append-eod', action='store_true',
    #                    help='Append an <eod> token to the end of a document.')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=32, help='Number of worker processes to launch')
    group.add_argument('--chunk-size', type=int, default=10, help='Chunk size assigned to each worker process')
    group.add_argument('--log-interval', type=int, default=1000, help='Interval between progress updates')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers)
    encoded_docs = pool.imap(encoder.encode, fin, args.chunk_size)
    # encoded_docs = map(encoder.encode, fin)

    assert Encoder.tokenizer.vocab_size < 65535
    print(f"Vocab size: {Encoder.tokenizer.vocab_size}")
    print(f"Output file: {args.input}.bin")
    fout = open(f'{args.input}.bin', 'wb')

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (tokens, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        if len(tokens) == 0:
            continue
        tokens = np.array(tokens, dtype=np.uint16)
        fout.write(tokens.tobytes(order='C'))
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents", f"({i/elapsed} docs/s, {mbs} MB/s).", file=sys.stderr)
    fout.close()


if __name__ == '__main__':
    main()
