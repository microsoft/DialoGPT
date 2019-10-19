#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 

import re
from util import *
from nltk.tokenize import TweetTokenizer

def clean_str(txt):
	#print("in=[%s]" % txt)
	txt = txt.lower()
	txt = re.sub('^',' ', txt)
	txt = re.sub('$',' ', txt)

	# url and tag
	words = []
	for word in txt.split():
		i = word.find('http') 
		if i >= 0:
			word = word[:i] + ' ' + '__url__'
		words.append(word.strip())
	txt = ' '.join(words)

	# remove markdown URL
	txt = re.sub(r'\[([^\]]*)\] \( *__url__ *\)', r'\1', txt)

	# remove illegal char
	txt = re.sub('__url__','URL',txt)
	txt = re.sub(r"[^A-Za-z0-9():,.!?\"\']", " ", txt)
	txt = re.sub('URL','__url__',txt)	

	# contraction
	add_space = ["'s", "'m", "'re", "n't", "'ll","'ve","'d","'em"]
	tokenizer = TweetTokenizer(preserve_case=False)
	txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '
	txt = txt.replace(" won't ", " will n't ")
	txt = txt.replace(" can't ", " can n't ")
	for a in add_space:
		txt = txt.replace(a+' ', ' '+a+' ')

	txt = re.sub(r'^\s+', '', txt)
	txt = re.sub(r'\s+$', '', txt)
	txt = re.sub(r'\s+', ' ', txt) # remove extra spaces
	
	#print("out=[%s]" % txt)
	return txt


if __name__ == '__main__':
	ss = [
		" I don't know:). how about this?https://github.com/golsun/deep-RL-time-series",
		"please try [ GitHub ] ( https://github.com )",
		]
	for s in ss:
		print(s)
		print(clean_str(s))
		print()
	
