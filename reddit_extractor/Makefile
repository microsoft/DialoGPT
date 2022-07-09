#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 

.SECONDARY:

include configs/Makefile.local

ifeq ($(SIZE), full)
all: train refs
else
all: train 
endif
refs: ../data/test.refs.txt
train: ../data/train.tsv
test: $T/test.tsv.gz

#### Merged files:

../data/test.refs.txt: ../data/test.scored_refs.txt
	sed 's/[0-9]\+|//g' < $+ > $@

../data/test.scored_refs.txt: $T/test.tsv.gz
	python src/create-multiref.py --data $< --testids data/test-multi-refs-ids.txt --out $@

../data/%.tsv: $T/%.tsv.gz
	zcat $+ | cut -f 2-3 > $@

$T/train.tsv.gz: $(TARGETS_TRAIN)
	zcat $(TARGETS_TRAIN) | gzip > $@

$T/test.tsv.gz: $(TARGETS_TEST)
	zcat $(TARGETS_TEST) | gzip > $@

#### Create extracts by month:

$T/test/$A/%.tsv.gz: $T/test/$P/%/stat.tsv prep
	mkdir -p $T/test/$A
	python src/reddit.py $(*F) --task=conv --keep_keys=data/keys-test.gz --parallel=True --reddit_input $S --reddit_output $T/test --clean True --min_score $(MIN_SCORE) --min_depth $(MIN_DEPTH) --max_depth $(MAX_DEPTH) --use_title $(TITLE) --leaves_only 0 > $@.log 2>&1
	gzip -f $T/test/$A/$(*F).tsv

$T/train/$A/%.tsv.gz: $T/train/$P/%/stat.tsv prep
	mkdir -p $T/train/$A
	python src/reddit.py $(*F) --task=conv $(HASH_FLAG) --keep_keys=$K/$(*F).gz --discard_tgt_keys $K/$(*F)-o.gz --parallel=True $(WORDS_BLOCKLIST) $(SUBREDDITS_BLOCKLIST) --reddit_input $S --reddit_output $T/train --clean True --min_score $(MIN_SCORE) --min_depth $(MIN_DEPTH) --max_depth $(MAX_DEPTH) --use_title $(TITLE) --leaves_only $(LEAVES_ONLY) > $@.log 2>&1
	gzip -f $T/train/$A/$(*F).tsv

$T/test/$P/%/stat.tsv: $S/RS_%.zst $S/RC_%.zst prep
	mkdir -p $T/test/$P/$(*F)
	python src/reddit.py $(*F) --keep_keys=data/keys-test.gz --task=extract --reddit_input $S --reddit_output $T/test > $@.log 2>&1
	gzip -f $T/test/$P/$(*F)/rc*.tsv
	gzip -f $T/test/$P/$(*F)/rs*.tsv

$T/train/$P/%/stat.tsv: $S/RS_%.zst $S/RC_%.zst prep
	mkdir -p $T/train/$P/$(*F)
	python src/reddit.py $(*F) --keep_keys=$K/$(*F).gz --task=extract --reddit_input $S --reddit_output $T/train > $@.log 2>&1
	gzip -f $T/train/$P/$(*F)/rc*.tsv
	gzip -f $T/train/$P/$(*F)/rs*.tsv

#### Download Reddit dumps:

$S/RS_%.zst: lists/files/RS_%.zst
	wget $(WARGS) $U/submissions/RS_$(*F).zst -O $S/RS_$(*F).zst -o logs/RS_$(*F).zst.log -c
$S/RC_%.zst: lists/files/RC_%.zst
	wget $(WARGS) $U/comments/RC_$(*F).zst -O $S/RC_$(*F).zst -o logs/RC_$(*F).zst.log -c

$S/RS_%.bz2: lists/files/RS_%.bz2
	wget $(WARGS) $U/submissions/RS_$(*F).bz2 -O $S/RS_$(*F).bz2 -o logs/RS_$(*F).bz2.log -c
	touch $@
$S/RC_%.bz2: lists/files/RC_%.bz2
	wget $(WARGS) $U/comments/RC_$(*F).bz2 -O $S/RC_$(*F).bz2 -o logs/RC_$(*F).bz2.log -c
	touch $@

#### Extraction preparation:

prep: $(LIST_REDDIT) lists/files/.create

$(LIST_REDDIT): lists/files/.create
	touch $@
lists/files/.create: $K/.create
	mkdir -p $S logs 
	mkdir -p lists/files
	touch $@

$K/.create: data/keys-$(SIZE).tar
	mkdir $K
	touch $@
	tar xvf $<

data/keys-full.tar:
	mkdir -p logs
	wget $(WARGS) https://acvrpublicycchen.blob.core.windows.net/dialogpt/keys-full.tar -O data/keys-full.tar
