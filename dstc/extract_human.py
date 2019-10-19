#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
#  Evaluate DSTC-task2 submissions. https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling

from tokenizers import *

key_file = "./data/processed/test_real.keys.txt"

human_hyp = "human.resp.txt"

refs = "./data/test.refs.txt"

all_keys = []

with open(key_file, 'r') as keys:
	for k in iter(keys):
		all_keys.append(k)

all_lines = {}
with open(refs, 'r', encoding='utf-8') as all_refs:
	for i,r in enumerate(all_refs):
		# import pdb; pdb.set_trace()
		key = r.split('\t')[0]
		try:	
			line = r.split('\t')[2].split('|')[1] #clean_str()
			all_lines[key] = line
		except:
			print(key)
			pass

			
with open(human_hyp, 'w') as f:	
	for key in all_keys:
		f.write(all_lines[key.strip()] + u'\n')	


