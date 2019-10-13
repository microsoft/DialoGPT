#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import os
from . import proj_env

RAW_DATA_DIR = os.path.join(proj_env.ROOT_DIR, "raw_data")
PROCESSED_DATA_DIR = os.path.join(proj_env.ROOT_DIR, "processed")
PIPELINE_DATA_DIR = os.path.join(proj_env.ROOT_DIR, "pipeline_data")

TEST_KEY_FN = os.path.join(RAW_DATA_DIR, "keys.2k.txt")


MAX_LEN = 128 #512
MAX_CONTEXT_LEN = 64#250

TAG_LIST = ["<p>", "<title>", "<anchor>"] + ["<h%s>" % i for i in range(1, 7)]

