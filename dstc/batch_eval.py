#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import subprocess as sp
import os
import argparse
import glob

CODE_ROOT = "./"
PYTHON_EXE = "python"


def eval(prediction_path, output_path, data_type):
    os.chdir(CODE_ROOT)
    ref_path = f'{CODE_ROOT}/data/test.refs.txt'
    key_path = f'{CODE_ROOT}/data/keys.2k.txt'
    cmd = ['dstc.py',
           prediction_path,
           '--refs',
           ref_path,
           '--keys',
           key_path,
           '--clean']
    cmd = " ".join(cmd) #% {"CODE_ROOT": CODE_ROOT}
    print(cmd)

    ret = sp.run([PYTHON_EXE] + cmd.split(" "), stdout=sp.PIPE, stderr=sp.STDOUT)

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(ret.stdout.decode("utf-8"))


parser = argparse.ArgumentParser(description="""
Given `input_dir`, create eval directory under it.
for each *.predicted.txt under `input_dir`, evaluate it, push the evaluation result to files under eval directory.
The file name is *.eval.txt
""")
parser.add_argument("--input_dir", type=str, default="./")
parser.add_argument("--data_type", type=str, default="test", help="could be 'valid' or 'test' ")

args = parser.parse_args()
assert args.data_type in ("valid", "test")

output_dir = os.path.join(args.input_dir, "eval")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

resp_dir = os.path.join(args.input_dir, "resp")
if not os.path.exists(resp_dir):
    os.makedirs(resp_dir)

for prediction_path in glob.glob(os.path.join(args.input_dir, "*.resp.txt")):
    resp_path = os.path.join(resp_dir , os.path.basename(prediction_path))
    cmd = ['less',
           prediction_path,
           '|grep -v "^Val"',
           '|sed \'/^$/d\'',
           '>',
           resp_path]
    cmd = " ".join(cmd)
    print(cmd)
    ret = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    output = ret.communicate()[0]
    print(output)
    output_name = os.path.basename(prediction_path).replace(".resp.txt", ".eval.txt")
    output_path = os.path.join(output_dir, output_name)
    eval(resp_path, output_path, args.data_type)

