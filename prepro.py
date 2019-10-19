#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
"""
preprocess input data into feature and stores binary as python shelve DB
each chunk is gzipped JSON string
"""
import argparse
import gzip
import json
import subprocess as sp
import shelve
import os
from os.path import dirname, exists, join

import torch
from lsp_model import GPT2Tokenizer
from tqdm import tqdm

from env import END_OF_TEXT_TOKEN
from gpt2_training.train_utils import InputFeatures_train as InputFeatures


def _get_file_len(corpus):
    n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                 universal_newlines=True).split()[0])
    return n_line


def _norm_text(text):
    w, *toks = text.strip().split()
    try:
        w = float(w)
    except Exception:
        toks = [w] + toks
        w = 1.0
    return w, ' '.join(toks)


def _get_inputs_from_text(text, tokenizer):
    srcs, tgt = text.strip().split('\t')
    weights = []
    inputs = []
    for src in srcs.split(' EOS '):
        src_weight, src = _norm_text(src)
        context_id = tokenizer.encode(src)
        weights.append(src_weight)
        inputs.append(context_id)
    tgt_weight, tgt = _norm_text(tgt)
    if tgt_weight != 0:
        response_id = tokenizer.encode(tgt)
        weights.append(tgt_weight)
        inputs.append(response_id)
    return weights, inputs


def _make_features(id_, weights, inputs, tokenizer, max_len):
    end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
    features = []
    sents = []
    ws = []
    len_ = 0
    i = 0
    for ids, w in zip(inputs, weights):
        if len(ids) > max_len:
            if len(sents) >= 2:
                feat = _make_feature(id_ + i, sents, ws, end_of_text_id)
                if feat is not None:
                    features.append(feat)
                    i += 1
            len_ = 0
            sents = []
            ws = []
            continue
        elif len_ > max_len:
            feat = _make_feature(id_ + i, sents, ws, end_of_text_id)
            if feat is not None:
                features.append(feat)
                i += 1
            len_ = len(sents[-1]) + 1
            sents = sents[-1:]
            ws = ws[-1:]
        len_ += (len(ids) + 1)
        sents.append(ids)
        ws.append(w)
    if len(sents) >= 2:
        feat = _make_feature(id_ + i, sents, ws, end_of_text_id)
        if feat is not None:
            features.append(feat)

    return features


def _make_feature(id_, sents, ws, eos):
    if all(w == 0 for w in ws[1:]):
        return None
    input_ids = [i for s in sents for i in s+[eos]][:-1]
    lm_labels = []
    weights = []
    token_type_ids = []  # this becomes round ids
    for i, (s, w) in enumerate(zip(sents, ws)):
        if i == 0:
            lm_labels += [-1] * len(s)
            weights += [0.0] * len(s)
            token_type_ids += [0] * len(s)
            continue

        token_type_ids += [i] * (len(s) + 1)
        if w == 0.0:
            lm_labels += [-1] * (len(s) + 1)
            weights += [0.0] * (len(s) + 1)
        else:
            lm_labels += (s + [eos])
            weights += [w] * (len(s) + 1)

    # handle trailing -1's
    i = len(lm_labels) - 1
    while i >= 0:
        if lm_labels[i] != -1:
            break
        i -= 1
    input_ids = input_ids[:i+1]
    lm_labels = lm_labels[:i+1]
    weights = weights[:i+1]
    token_type_ids = token_type_ids[:i+1]

    # pad to multiples of 8
    while len(input_ids) % 8 != 0:
        input_ids.append(0)
        token_type_ids.append(0)
        lm_labels.append(-1)
        weights.append(0.0)

    position_ids = list(range(len(input_ids)))
    assert (len(input_ids) == len(position_ids) == len(token_type_ids)
            == len(lm_labels) == len(weights))
    assert len(input_ids) % 8 == 0
    if len(input_ids) == 0:
        import pdb
        pdb.set_trace()
    feature = InputFeatures(id_, input_ids, position_ids, token_type_ids,
                            lm_labels, weights)
    return feature


def main(args):
    toker = GPT2Tokenizer.from_pretrained('gpt2')
    attrs = []
    if args.reverse:
        attrs.append('reverse')
    if args.two_turn:
        attrs.append('2turn')
    if attrs:
        db_path = (f'{args.corpus[:-4]}.{args.max_seq_len}len.'
                   f'{".".join(attrs)}.db/db')
    else:
        db_path = f'{args.corpus[:-4]}.{args.max_seq_len}len.db/db'
    if exists(dirname(db_path)):
        raise ValueError('Found existing DB, please backup')
    else:
        os.makedirs(dirname(db_path))
    with open(args.corpus, "r", encoding="utf-8") as reader, \
            shelve.open(db_path, 'n') as db:
        chunk = []
        n_chunk = 0
        n_example = 0
        for line in tqdm(reader, total=_get_file_len(args.corpus)):
            try:
                if len(chunk) >= args.chunk_size:
                    # save and renew chunk
                    db[f'chunk_{n_chunk}'] = gzip.compress(
                        json.dumps(chunk[:args.chunk_size]).encode('utf-8'))
                    chunk = chunk[args.chunk_size:]
                    n_chunk += 1

                weights, inputs = _get_inputs_from_text(line, toker)
                if args.reverse:
                    weights = list(reversed(weights))
                    inputs = list(reversed(inputs))
                if args.two_turn:
                    weights = weights[:2]
                    inputs = inputs[:2]
                if len(weights) < 2:
                    continue
                features = _make_features(n_example, weights, inputs,
                                          toker, args.max_seq_len)
                for feature in features:
                    chunk.append(vars(feature))
                    n_example += 1
            except Exception as e:
                print('!!! prepro exception !!!', e)
                continue
        # save last chunk
        db[f'chunk_{n_chunk}'] = gzip.compress(
            json.dumps(chunk).encode('utf-8'))
    # save relevant information to reproduce
    meta = {'n_example': n_example,
            'chunk_size': args.chunk_size,
            'max_seq_len': args.max_seq_len,
            'reverse': args.reverse,
            'two_turn': args.two_turn}
    with open(join(dirname(db_path), 'meta.json'), 'w') as writer:
        json.dump(meta, writer, indent=4)
    torch.save(toker, join(dirname(db_path), 'tokenizer.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='file name of training corpus (should be .tsv)')
    parser.add_argument('--chunk_size', type=int, default=65536,
                        help='num of data examples in a storing chunk')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='discard data longer than this')
    parser.add_argument('--reverse', action='store_true',
                        help='reverse the src tgt')
    parser.add_argument('--two_turn', action='store_true',
                        help='take only the first 2 turns')

    args = parser.parse_args()

    main(args)
