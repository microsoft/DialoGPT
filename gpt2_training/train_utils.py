#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 

import os
import logging
import torch
from collections import defaultdict

from env import END_OF_TEXT_TOKEN
from lsp_model.optim import warmup_linear, noam_decay, noamwd_decay


logger = logging.getLogger(__name__)

SEQ_LENGTH_SHRINK_PROP = 0.9


def load_model(model, checkpoint, args, verbose=False):
    n_gpu = args.n_gpu
    device = args.device
    if checkpoint is None or checkpoint == "None":
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if (hasattr(model, "transformer")
            and all(not s.startswith('transformer.')
                    for s in model_state_dict.keys())):
            logger.info('loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict)

    if args.fp16:
        logger.info('in fp16, model.half() activated')
        model.half()
    model.to(device)
    if n_gpu > 1:
        logging.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict


class InputFeatures(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 lm_labels, context_len, response_len):
        self.conv_id = conv_id
        self.choices_features = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        self.lm_labels = lm_labels
        self.context_len = context_len
        self.response_len = response_len    # in case we need it


class InputFeatures_train(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 lm_labels, weights, input_len=None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.lm_labels = lm_labels
        self.weights = weights
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len


class RedditExample(object):
    def __init__(self, conv_id, context, response):
        self.conv_id = conv_id
        self.context = context
        self.response = response

    def __repr__(self):
        return 'conv_id = {}\ncontext = {}\nresponse = {}'.format(
            self.conv_id, self.context, self.response)

    def __str__(self):
        return self.__repr__()


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def get_eval_list_same_length(input_file, tokenizer, max_batch_size,
                              norm=True):
    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        content = [l.split('\t') for l in f.read().splitlines()]

    context, response = [c[0] for c in content], [c[1:] for c in content]
    i = 0
    for src, tgt_all in zip(context, response):
        for tgt in tgt_all:
            if norm:
                src_line = ' '.join(src.strip().split())
                tgt_line = ' '.join(tgt.strip().split())
            else:
                src_line = src.strip()
                tgt_line = tgt.strip()
            examples.append(RedditExample(i, src_line, tgt_line))
            i += 1

    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        response_id = tokenizer.encode(example.response)
        input_ids = context_id + [end_of_text_id]
        lm_labels = response_id

        position_ids = list(range(len(input_ids)))

        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(context_id), len(response_id))

    def batch_feature_same_len(features):
        input_ids = torch.stack([torch.tensor(f.choices_features['input_ids'],
                                              dtype=torch.long)
                                 for f in features])
        position_ids = torch.stack(
            [torch.tensor(f.choices_features['position_ids'], dtype=torch.long)
             for f in features])
        token_type_ids = torch.stack(
            [torch.tensor(f.choices_features['token_type_ids'],
                          dtype=torch.long)
             for f in features])
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f.lm_labels, dtype=torch.long) for f in features],
            batch_first=True, padding_value=-1)

        context_len = torch.tensor([f.context_len for f in features],
                                   dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features],
                                    dtype=torch.long)
        return (input_ids, position_ids, token_type_ids, labels,
                context_len, response_len)

    features = [featurize(e) for e in examples]
    dataloader_pre = defaultdict(list)
    for f in features:
        dataloader_pre[f.context_len].append(f)

    dataloader = []
    for l in sorted(dataloader_pre):
        f = batch_feature_same_len(dataloader_pre[l])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size]
                                   for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]):
                    break
    return dataloader


def set_lr(optimizer, step, schedule, lr,
           warmup_steps, warmup_proportion, n_embd, tot_steps):
    if schedule == 'None':
        lr_this_step = lr
    elif schedule == 'noam':  # transformer like
        lr_this_step = lr * 1e4 * noam_decay(step+1, warmup_steps, n_embd)
    elif schedule == 'noamwd':  # transformer like
        lr_this_step = lr * 1e4 * noamwd_decay(step+1, warmup_steps, n_embd)
    else:
        lr_this_step = lr * warmup_linear(step / tot_steps,
                                          warmup_proportion)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step
