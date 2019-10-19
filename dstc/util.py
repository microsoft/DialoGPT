#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import os, time, subprocess, io, sys, re, argparse
import numpy as np

py_version = sys.version.split('.')[0]
if py_version == '2':
	open = io.open
else:
	unicode = str

def makedirs(fld):
	if not os.path.exists(fld):
		os.makedirs(fld)


def str2bool(s):
	# to avoid issue like this: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	if s.lower() in ['t','true','1','y']:
		return True
	elif s.lower() in ['f','false','0','n']:
		return False
	else:
		raise ValueError