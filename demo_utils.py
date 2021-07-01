#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

import os
import logging

from pytorch_pretrained_bert.file_utils import http_get


logger = logging.getLogger(__name__)


# Note that the model size is roughly half of the GPT model because our model is saved by fp16
LSP_MODEL_URL = {
    'multiref': {
        'large_fs': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/multiref/large_fs.pkl',
        'medium_fs': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/multiref/medium_fs.pkl',
        'medium_ft': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/multiref/medium_ft.pkl',
        'small_fs': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/multiref/small_fs.pkl',
        'small_ft': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/multiref/small_ft.pkl'
    },
    'dstc': {
        'medium_ft': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/DSTC/medium_ft.pkl'
    }
}

# GPT model could be downloaded from huggingface repo
GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "small": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin"
}

CONFIG_FILE = {
    'small': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/117M/config.json',
    'medium': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/345M/config.json',
    'large': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/1542M/config.json'
}

VOCAB_FILE = {
    'small': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/117M/vocab.json',
    'medium': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/345M/vocab.json',
    'large': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/1542M/vocab.json'
}

MERGE_FILE = {
    'small': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/117M/merges.txt',
    'medium': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/345M/merges.txt',
    'large': 'https://acvrpublicycchen.blob.core.windows.net/dialogpt/1542M/merges.txt'
}


def download_file(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    file_name = os.path.basename(url)
    if 'pytorch_model.bin' in file_name:
        file_name = 'pytorch_model.bin'

    if os.path.isfile(os.path.join(folder, file_name)):
        logger.info(f'{os.path.join(folder, file_name)} exists, return!')
        return

    with open(os.path.join(folder, file_name), 'wb') as f:
        http_get(url, f)


def download_model_folder(model_size, dataset=None, from_scratch=None, DATA_FOLDER=None):
    assert DATA_FOLDER is not None, 'DATA_FOLDER cannot be None'
    assert model_size in ['small', 'medium', 'large'], 'model size should be one of \'small\', \'medium\' or \'large\''
    target_folder = os.path.join(DATA_FOLDER, model_size)
    download_file(CONFIG_FILE[model_size], target_folder)
    download_file(VOCAB_FILE[model_size], target_folder)
    download_file(MERGE_FILE[model_size], target_folder)
    download_file(GPT2_PRETRAINED_MODEL_ARCHIVE_MAP[model_size], target_folder)
    if dataset is not None:
        assert dataset in ['multiref', 'dstc'], \
            'dataset has to be \'multiref\' or \'dstc\''
        assert from_scratch in [True, False], 'from scratch has to be True or False'

        if from_scratch:
            model_train_type = model_size + '_fs'
        else:
            model_train_type = model_size + '_ft'
        if model_train_type not in LSP_MODEL_URL[dataset]:
            k = ','.join(list(LSP_MODEL_URL[dataset].keys()))
            raise ValueError(f'\'{model_train_type}\' not exist for dataset \'{dataset}\', please choose from [{k}]')
        download_file(LSP_MODEL_URL[dataset][model_train_type], target_folder)
    return target_folder

