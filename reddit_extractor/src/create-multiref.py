#!/usr/bin/env python
#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 

import sys
import gzip
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data", required=True, help="gz file containing test data")
parser.add_argument("--testids", required=True, help="multi-ref test set with string replaced with IDs")
parser.add_argument("--out", required=True, help="output multi-ref file")

args = parser.parse_args()

data = {}

with gzip.open(args.data, 'rt', encoding="utf-8") as f:
	for line in f:
		line = line.rstrip()
		line = line.replace('“','"')
		line = line.replace('”','"')
		keys, src, tgt = line.split('\t')
		data[keys] = (src, tgt)

with open(args.out, 'wt', encoding="utf-8") as fo:
	with open(args.testids, 'rt', encoding="utf-8") as fi:
		for line in fi:
			line = line.rstrip()
			els = line.split('\t')
			header = els[0]
			if header != 'multiref':
				print("Ignoring line: " + line, file=sys.stderr)
			rscore1, rids1 = els[1].split(',', 1)
			if rids1 not in data.keys():
				print("Error: can't find data for ref ID: %s" % rids1, file=sys.stderr)
				continue
			src, r1 = data[rids1]
			scores = [ rscore1 ]
			refs = [ r1 ]
			for el in els[2:]:
				rscoreI, ridsI = el.split(',', 1)
				if ridsI not in data.keys():
					print("Error: can't find data for ref ID: %s" % ridsI, file=sys.stderr)
				else:
					srcI, rI = data[ridsI]
					if srcI != src:
						print("Error: mismatch source for ref ID: %s" % ridsI, file=sys.stderr)
					else:
						scores.append(rscoreI)
						refs.append(rI)

			# Write multi-ref instance:
			fo.write('%s' % src)
			for i in range(len(scores)):
				fo.write('\t%s|%s' % (scores[i], refs[i]))
			fo.write('\n')
