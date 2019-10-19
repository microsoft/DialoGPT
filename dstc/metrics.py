#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 

import re
from util import *
from collections import defaultdict


def calc_nist_bleu(path_refs, path_hyp, fld_out='temp', n_lines=None):
	# call mteval-v14c.pl
	# ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v14c.pl
	# you may need to cpan install XML:Twig Sort:Naturally String:Util 

	makedirs(fld_out)

	if n_lines is None:
		n_lines = len(open(path_refs[0], encoding='utf-8').readlines())	
	# import pdb; pdb.set_trace()
	_write_xml([''], fld_out + '/src.xml', 'src', n_lines=n_lines)
	_write_xml([path_hyp], fld_out + '/hyp.xml', 'hyp')#, n_lines=n_lines)
	_write_xml(path_refs, fld_out + '/ref.xml', 'ref')#, n_lines=n_lines)

	time.sleep(1)
	cmd = [
		'perl','3rdparty/mteval-v14c.pl',
		'-s', '%s/src.xml'%fld_out,
		'-t', '%s/hyp.xml'%fld_out,
		'-r', '%s/ref.xml'%fld_out,
		]
	process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	# import pdb; pdb.set_trace()
	output, error = process.communicate()

	lines = output.decode().split('\n')

	try:
		nist = lines[-6].strip('\r').split()[1:5]
		bleu = lines[-4].strip('\r').split()[1:5]
		return [float(x) for x in nist], [float(x) for x in bleu]

	except Exception:
		print('mteval-v14c.pl returns unexpected message')
		print('cmd = '+str(cmd))
		print(output.decode())
		print(error.decode())
		return [-1]*4, [-1]*4

	


def calc_cum_bleu(path_refs, path_hyp):
	# call multi-bleu.pl
	# https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
	# the 4-gram cum BLEU returned by this one should be very close to calc_nist_bleu
	# however multi-bleu.pl doesn't return cum BLEU of lower rank, so in nlp_metrics we preferr calc_nist_bleu
	# NOTE: this func doesn't support n_lines argument and output is not parsed yet

	process = subprocess.Popen(
			['perl', '3rdparty/multi-bleu.perl'] + path_refs, 
			stdout=subprocess.PIPE, 
			stdin=subprocess.PIPE
			)
	with open(path_hyp, encoding='utf-8') as f:
		lines = f.readlines()
	for line in lines:
		process.stdin.write(line.encode())
	output, error = process.communicate()
	return output.decode()


def calc_meteor(path_refs, path_hyp, fld_out='temp', n_lines=None, pretokenized=True):
	# Call METEOR code.
	# http://www.cs.cmu.edu/~alavie/METEOR/index.html

	makedirs(fld_out)
	path_merged_refs = fld_out + '/refs_merged.txt'
	_write_merged_refs(path_refs, path_merged_refs)
	cmd = [
			'java', '-Xmx1g',	# heapsize of 1G to avoid OutOfMemoryError
			'-jar', '3rdparty/meteor-1.5/meteor-1.5.jar', 
			path_hyp, path_merged_refs, 
			'-r', '%i'%len(path_refs), 	# refCount 
			'-l', 'en', '-norm' 	# also supports language: cz de es fr ar
			]
	print(cmd)
	process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()
	for line in output.decode().split('\n'):
		if "Final score:" in line:
			return float(line.split()[-1])

	print('meteor-1.5.jar returns unexpected message')
	print("cmd = " + " ".join(cmd))
	print(output.decode())
	print(error.decode())
	return -1 


def calc_entropy(path_hyp, n_lines=None):
	# based on Yizhe Zhang's code
	etp_score = [0.0,0.0,0.0,0.0]
	counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
	i = 0
	for line in open(path_hyp, encoding='utf-8'):
		i += 1
		words = line.strip('\n').split()
		for n in range(4):
			for idx in range(len(words)-n):
				ngram = ' '.join(words[idx:idx+n+1])
				counter[n][ngram] += 1
		if i == n_lines:
			break

	for n in range(4):
		total = sum(counter[n].values())
		for v in counter[n].values():
			etp_score[n] += - v /total * (np.log(v) - np.log(total))

	return etp_score


def calc_len(path, n_lines):
	l = []
	for line in open(path, encoding='utf8'):
		l.append(len(line.strip('\n').split()))
		if len(l) == n_lines:
			break
	return np.mean(l)


def calc_diversity(path_hyp):
	tokens = [0.0,0.0]
	types = [defaultdict(int),defaultdict(int)]
	for line in open(path_hyp, encoding='utf-8'):
		words = line.strip('\n').split()
		for n in range(2):
			for idx in range(len(words)-n):
				ngram = ' '.join(words[idx:idx+n+1])
				types[n][ngram] = 1
				tokens[n] += 1
	div1 = len(types[0].keys())/tokens[0]
	div2 = len(types[1].keys())/tokens[1]
	return [div1, div2]


def nlp_metrics(path_refs, path_hyp, fld_out='temp',  n_lines=None):
	nist, bleu = calc_nist_bleu(path_refs, path_hyp, fld_out, n_lines)
	meteor = calc_meteor(path_refs, path_hyp, fld_out, n_lines)
	entropy = calc_entropy(path_hyp, n_lines)
	div = calc_diversity(path_hyp)
	avg_len = calc_len(path_hyp, n_lines)
	return nist, bleu, meteor, entropy, div, avg_len


def _write_merged_refs(paths_in, path_out, n_lines=None):
	# prepare merged ref file for meteor-1.5.jar (calc_meteor)
	# lines[i][j] is the ref from i-th ref set for the j-th query

	lines = []
	for path_in in paths_in:
		lines.append([line.strip('\n') for line in open(path_in, encoding='utf-8')])

	with open(path_out, 'w', encoding='utf-8') as f:
		for j in range(len(lines[0])):
			for i in range(len(paths_in)):
				f.write(unicode(lines[i][j]) + "\n")



def _write_xml(paths_in, path_out, role, n_lines=None):
	# prepare .xml files for mteval-v14c.pl (calc_nist_bleu)
	# role = 'src', 'hyp' or 'ref'

	lines = [
		'<?xml version="1.0" encoding="UTF-8"?>',
		'<!DOCTYPE mteval SYSTEM "">',
		'<!-- generated by https://github.com/golsun/NLP-tools -->',
		'<!-- from: %s -->'%paths_in,
		'<!-- as inputs for ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v14c.pl -->',
		'<mteval>',
		]

	for i_in, path_in in enumerate(paths_in):

		# header ----

		if role == 'src':
			lines.append('<srcset setid="unnamed" srclang="src">')
			set_ending = '</srcset>'
		elif role == 'hyp':
			lines.append('<tstset setid="unnamed" srclang="src" trglang="tgt" sysid="unnamed">')
			set_ending = '</tstset>'
		elif role == 'ref':
			lines.append('<refset setid="unnamed" srclang="src" trglang="tgt" refid="ref%i">'%i_in)
			set_ending = '</refset>'
		
		lines.append('<doc docid="unnamed" genre="unnamed">')

		# body -----

		if role == 'src':
			body = ['__src__'] * n_lines
		else:
			with open(path_in, 'r', encoding='utf-8') as f:
				body = f.readlines()
			if n_lines is not None:
				body = body[:n_lines]
		#for i in range(len(body)):
		i = 0
		for b in body:
			line = b.strip('\n')
			line = line.replace('&',' ').replace('<',' ')		# remove illegal xml char
			# if len(line) > 0:
			lines.append('<p><seg id="%i"> %s </seg></p>'%(i + 1, line))
			i += 1

		# ending -----

		lines.append('</doc>')
		if role == 'src':
			lines.append('</srcset>')
		elif role == 'hyp':
			lines.append('</tstset>')
		elif role == 'ref':
			lines.append('</refset>')

	lines.append('</mteval>')
	with open(path_out, 'w', encoding='utf-8') as f:
		f.write(unicode('\n'.join(lines)))
