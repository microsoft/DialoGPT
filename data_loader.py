#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import gzip
import json
import math
import random
import shelve
import torch

import subprocess as sp

from math import ceil
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence

from env import END_OF_TEXT_TOKEN
from gpt2_training.train_utils import (InputFeatures, InputFeatures_train,
                                       RedditExample)


class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class GPT2FeatureDataset(Dataset):
    """ pytorch dataset for GPT2 training """
    def __init__(self, features, max_len=None):
        self.features = features
        self.max_len = max_len  # this max_len do truncate

    def __getitem__(self, i):
        feat_dict = self.features[i]
        if self.max_len is not None and feat_dict['input_len'] > self.max_len:
            # tuncate on the left side (context)
            feat_dict['input_ids'] = feat_dict['input_ids'][-self.max_len:]
            feat_dict['position_ids'] = feat_dict['position_ids'][
                -self.max_len:]
            feat_dict['token_type_ids'] = feat_dict['token_type_ids'][
                -self.max_len:]
            feat_dict['lm_labels'] = feat_dict['lm_labels'][-self.max_len:]
        try:
            for s in ['context_len', 'response_len']:
                if s in feat_dict.keys():
                    print("db file missing "+s)
                    del feat_dict[s]
        except Exception:
            import pdb
            pdb.set_trace()

        feat = InputFeatures_train(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        return (input_ids, position_ids, token_type_ids, labels)


class BucketingDataLoader(object):
    """ this loads shelve db chunks and then convert to mini-batch loader"""
    def __init__(self, db_name, batch_size, max_seq_len,
                 bucket=100, shuffle=True):
        self.db = shelve.open(f'{db_name}/db', 'r')
        self.batch_size = batch_size
        self.max_len = max_seq_len
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle

    def _get_keys(self):
        keys = list(self.db.keys())
        return keys

    def __iter__(self):
        keys = self._get_keys()
        if self.shuffle:
            random.shuffle(keys)
        for key in keys:
            chunk = json.loads(gzip.decompress(self.db[key]).decode('utf-8'))
            # discard long examples
            trunc_chunk = []
            lens = []
            for feat in chunk:
                if feat['input_len'] > self.max_len:
                    continue
                trunc_chunk.append(feat)
                lens.append(feat['input_len'])

            dataset = GPT2FeatureDataset(trunc_chunk, self.max_len)
            sampler = BucketSampler(lens, self.bucket_size, self.batch_size,
                                    droplast=True, shuffle=self.shuffle)
            loader = DataLoader(dataset, batch_sampler=sampler,
                                num_workers=0,  # can test multi-worker
                                collate_fn=GPT2FeatureDataset.collate)
            yield from loader

    def __len__(self):
        raise NotImplementedError()

    def __del__(self):
        self.db.close()


class DistributedBucketingDataLoader(BucketingDataLoader):
    """ distributed version """
    def __init__(self, rank, num_replica, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.num_replica = num_replica

    def _get_keys(self):
        keys = list(self.db.keys())[self.rank::self.num_replica]
        return keys


def convert_examples_to_features_dynamic(examples, tokenizer,
                                         max_seq_length=512):
    """
    do not pad
    """
    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        # response is provided in example
        response_id = tokenizer.encode(example.response)

        input_ids_len = len(context_id) + len(response_id) + 2
        if input_ids_len > max_seq_length:
            if len(context_id) > input_ids_len - max_seq_length:
                # cut context from beginning if length of context + response is too long
                # and len of context is long enough to cut
                context_id = context_id[input_ids_len - max_seq_length:]
            else:
                # cut response from end if length of context + response is too long
                # and len of response is long enough to cut
                # if no response is available, discard the data
                if max_seq_length-len(context_id)-2 < 0:
                    return None
                response_id = response_id[:max_seq_length-len(context_id)-2]

        input_ids = context_id + [end_of_text_id] + response_id + [end_of_text_id]

        # label simplely is next token in sequences. MASK all context_id tokens except for the last one
        lm_labels = [-1] * len(context_id) + response_id + [end_of_text_id] + [-1]

        position_ids = list(range(len(input_ids)))

        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(context_id), len(response_id))

    # discard None feature
    features = [f for f in [featurize(ex) for ex in examples] if f is not None]
    return features


class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """
    def __init__(self, corpus_file, tokenizer, normalize_data,
                 batch_size, max_seq_length):
        self.corpus = corpus_file
        self.toker = tokenizer
        self.norm = normalize_data
        self.bs = batch_size
        self.max_seq_length = max_seq_length
        self.num_examples = self.get_len(corpus_file)

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples/self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as corpus:
                i = 0
                while True:
                    examples = []
                    cur_bs = 0
                    while True:
                        line = next(corpus).encode('utf-8').decode('utf-8')
                        contents = line.split('\t')
                        src, tgt_all = contents[0], contents[1:]
                        for tgt in tgt_all:
                            if self.norm:
                                src_line = ' '.join(src.strip().split())
                                tgt_line = ' '.join(tgt.strip().split())
                            else:
                                src_line = src.strip()
                                tgt_line = tgt.strip()
                            examples.append(
                                RedditExample(i, src_line, tgt_line),
                            )
                            i += 1
                            cur_bs += 1
                        if cur_bs >= self.bs:
                            break
                    features = convert_examples_to_features_dynamic(
                        examples, self.toker, self.max_seq_length)
                    batch = self._batch_feature(features)
                    yield batch
        except StopIteration:
            pass

    def _batch_feature(self, features):
        input_ids = pad_sequence([torch.tensor(f.choices_features['input_ids'],
                                               dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence(
            [torch.tensor(f.choices_features['position_ids'], dtype=torch.long)
             for f in features],
            batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(
            [torch.tensor(f.choices_features['token_type_ids'],
                          dtype=torch.long)
             for f in features],
            batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        context_len = torch.tensor([f.context_len for f in features],
                                   dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features],
                                    dtype=torch.long)
        return (input_ids, position_ids, token_type_ids, labels,
                context_len, response_len)

    def get_len(self, corpus):
        n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                     universal_newlines=True).split()[0])
        return n_line
