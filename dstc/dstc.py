#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
#  Evaluate DSTC-task2 submissions. https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling

from util import *
from metrics import *
from tokenizers import *

def extract_cells(path_in, path_hash):
	keys = [line.strip('\n') for line in open(path_hash)]
	cells = dict()
	for line in open(path_in, encoding='utf-8'):
		c = line.strip('\n').split('\t')
		k = c[0]
		if k in keys:
			cells[k] = c[1:]
	return cells

def extract_linc_cells(path_in, path_hash):
	if "valid" in path_hash:
		cells = dict()
		external_keys = [k.strip() for k in open(r"./data/processed/valid.keys.txt")]
		for no, line in enumerate(open(path_in, encoding='utf-8')):
			c = line.strip('\n')
			k = external_keys[no]
			cells[k] = [c]
	else:
		keys = set([line.strip('\n') for line in open(path_hash)])
		cells = dict()
		external_keys = [k.strip() for k in open(r"./data/processed/test_real.keys.txt")]
		for no, line in enumerate(open(path_in, encoding='utf-8')):
			c = line.strip('\n')
			k = external_keys[no]
			if k in keys:
				cells[k] = [c]
	return cells


def extract_hyp_refs(raw_hyp, raw_ref, path_hash, fld_out, n_refs=6, clean=False, vshuman=-1):
	cells_hyp = extract_linc_cells(raw_hyp, path_hash)
	cells_ref = extract_cells(raw_ref, path_hash)
	if not os.path.exists(fld_out):
		os.makedirs(fld_out)

	def _clean(s):
		if clean:
			return clean_str(s)
		else:
			return s

	keys = sorted(cells_hyp.keys())
	with open(fld_out + '/hash.txt', 'w', encoding='utf-8') as f:
		f.write(unicode('\n'.join(keys)))

	lines = [_clean(cells_hyp[k][-1]) for k in keys]
	path_hyp = fld_out + '/hyp.txt'
	with open(path_hyp, 'w', encoding='utf-8') as f:
		f.write(unicode('\n'.join(lines)))
	
	lines = []
	for _ in range(n_refs):
		lines.append([])
	for k in keys:
		refs = cells_ref[k]
		for i in range(n_refs):
			idx = i % len(refs)
			if idx == vshuman:
			    idx = (idx + 1) % len(refs)
			if "|" in refs[idx]:
				final_ref = refs[idx].split('|')[1]
			else:
				final_ref = refs[idx]
			lines[i].append(_clean(final_ref))

	path_refs = []
	for i in range(n_refs):
		path_ref = fld_out + '/ref%i.txt'%i
		with open(path_ref, 'w', encoding='utf-8') as f:
			f.write(unicode('\n'.join(lines[i])))
		path_refs.append(path_ref)

	return path_hyp, path_refs


def eval_one_system(submitted, keys, multi_ref, n_refs=6, n_lines=None, clean=False, vshuman=-1, PRINT=True):

	print('evaluating %s' % submitted)

	fld_out = submitted.replace('.txt','')
	if clean:
		fld_out += '_cleaned'
	path_hyp, path_refs = extract_hyp_refs(submitted, multi_ref, keys, fld_out, n_refs, clean=clean, vshuman=vshuman)
	nist, bleu, meteor, entropy, div, avg_len = nlp_metrics(path_refs, path_hyp, fld_out, n_lines=n_lines)
	
	if n_lines is None:
		n_lines = len(open(path_hyp, encoding='utf-8').readlines())

	if PRINT:
		print('n_lines = '+str(n_lines))
		print('NIST = '+str(nist))
		print('BLEU = '+str(bleu))
		print('METEOR = '+str(meteor))
		print('entropy = '+str(entropy))
		print('diversity = ' + str(div))
		print('avg_len = '+str(avg_len))

	return [n_lines] + nist + bleu + [meteor] + entropy + div + [avg_len]


def eval_all_systems(files, path_report, keys, multi_ref, n_refs=6, n_lines=None, clean=False, vshuman=False):
	# evaluate all systems (*.txt) in each folder `files`

	with open(path_report, 'w') as f:
		f.write('\t'.join(
				['fname', 'n_lines'] + \
				['nist%i'%i for i in range(1, 4+1)] + \
				['bleu%i'%i for i in range(1, 4+1)] + \
				['meteor'] + \
				['entropy%i'%i for i in range(1, 4+1)] +\
				['div1','div2','avg_len']
			) + '\n')

	for fl in files:
		if fl.endswith('.txt'):
			submitted = fl
			results = eval_one_system(submitted, keys=keys, multi_ref=multi_ref, n_refs=n_refs, clean=clean, n_lines=n_lines, vshuman=vshuman, PRINT=False)
			with open(path_report, 'a') as f:
				f.write('\t'.join(map(str, [submitted] + results)) + '\n')
		else:
			for fname in os.listdir(fl):
				if fname.endswith('.txt'):
					submitted = fl + '/' + fname
					results = eval_one_system(submitted, keys=keys, multi_ref=multi_ref, n_refs=n_refs, clean=clean, n_lines=n_lines, vshuman=vshuman, PRINT=False)
					with open(path_report, 'a') as f:
						f.write('\t'.join(map(str, [submitted] + results)) + '\n')

	print('report saved to: '+path_report, file=sys.stderr)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('submitted')	# if 'all' or '*', eval all teams listed in dstc/teams.txt
	                                    # elif endswith '.txt', eval this single file
	                                    # else, eval all *.txt in folder `submitted_fld`

	parser.add_argument('--clean', '-c', action='store_true')     # whether to clean ref and hyp before eval
	parser.add_argument('--n_lines', '-n', type=int, default=-1)  # eval all lines (default) or top n_lines (e.g., for fast debugging)
	parser.add_argument('--n_refs', '-r', type=int, default=6)    # number of references
	parser.add_argument('--vshuman', '-v', type=int, default='1') # when evaluating against human performance (N in refN.txt that should be removed) 
	                                                                      # in which case we need to remove human output from refs
	parser.add_argument('--refs', '-g', default='dstc/test.refs')
	parser.add_argument('--keys', '-k', default='keys/test.2k.txt')
	parser.add_argument('--teams', '-i', type=str, default='dstc/teams.txt')
	parser.add_argument('--report', '-o', type=str, default=None)
	args = parser.parse_args()
	print('Args: %s\n' % str(args), file=sys.stderr)

	if args.n_lines < 0:
		n_lines = None	# eval all lines
	else:
		n_lines = args.n_lines	# just eval top n_lines

	if args.submitted.endswith('.txt'):
		eval_one_system(args.submitted, keys=args.keys, multi_ref=args.refs, clean=args.clean, n_lines=n_lines, n_refs=args.n_refs, vshuman=args.vshuman)
	else:
		fname_report = 'report_ref%i'%args.n_refs
		if args.clean:
			fname_report += '_cleaned'
		fname_report += '.tsv'
		if args.submitted == 'all' or args.submitted == '*':
			files = ['dstc/' + line.strip('\n') for line in open(args.teams)]
			path_report = 'dstc/' + fname_report
		else:
			files = [args.submitted]
			path_report = args.submitted + '/' + fname_report
		if args.report != None:
			path_report = args.report
		eval_all_systems(files, path_report, keys=args.keys, multi_ref=args.refs, clean=args.clean, n_lines=n_lines, n_refs=args.n_refs, vshuman=args.vshuman)
