# A State-of-the-Art Large-scale Pretrained Response generation model (DialogLSP)

This repo contains the source code and trained model for a large-scale pretrained dialogue response generation model. See more details on our [project page](https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation/)

The repo is based on [huggingface pytorch-transformer](https://github.com/huggingface/transfer-learning-conv-ai) and [OpenAI GPT-2](https://github.com/openai/gpt-2), containing data extraction script, model training code and pretrained small (117M) medium (345M) and large (762M) model checkpoint.

The model is trained on 147M multi-turn dialogue from Reddit discussion thread. The large model can be trained in about *50-100 hours on a 8 V100 machine*, with distributed training and FP16 option. 

The include script can be used to reproduce the results of DSTC-7 grounded dialogue generation challenge and a *6k multi-reference dataset* created from Reddit data. 


**This github repository will be updated soon. Please stay tuned.**




## Setup & Installation

Please use the below commandlines to clone and install *the requirements:*

```bash
git clone https://github.com/microsoft/DialogLSP.git
cd DialogLSP
pip install -r requirements.txt
```

## Train the model with Docker environment

The image environment for running the code can be *loaded as below*:  

```bash
$ docker run --rm -it icaruszyz/large-scale-training:v5 bash
```

You can then *run* the `bash load_model.sh` to get a pretrained model:

```bash
bash load_model.sh
```

## Pretrained model

The pretrained and fine-tuned model are available on azure blobstorage [here](). The `load_model.sh` will automatically download and use the pretrained model. 

## Preparing data

The trainig data need to be first processed into a database file with below *commandline*

```bash
python prepro_v4.py
```

## Using the training script

The training script can be used in single GPU or multi GPU settings:

```bash
python ./LSP_train.py  # Single GPU training
python -m torch.distributed.launch --nproc_per_node=8 ./LSP_train.py  # Training on 8 GPUs
```


The training script accept *several arguments* to tweak the training: 

Argument | Type | Default value | Description
---------|------|---------------|------------
max\_seq\_length | `int` | `128` | Maximum number of tokens for each training instance. 
train\_input\_file | `str` | `""` | Path of the training dataset in a .db format
eval\_input\_file | `str` | `""` | Path of the validation set in a tsv format
continue_from | `int` | `0` | Resuming the training after a specified number of steps
fp16 | `boolean` | `True` | Whether to use 16-bits floating point for model training.
train\_batch\_size | `int` | `4` | Batch size for training
valid\_batch\_size | `int` | `4` | Batch size for validation
gradient\_accumulation\_steps | `int` | `2` | Accumulate gradients on several steps
learning\_rate | `float` | `1e-5` | Learning rate
lr\_schedule | `str` | `noam` | Learning rate schedule can be chosen from [`noam`, `noamwd`, `BERT`, `None`]
num\_optim\_steps | `int` | `1000000` | Number of training optimization steps


## DSTC challenge

Here is how to reproduce our DSTC results on a server with 8 V100 GPUs (adapt number of nodes and batch sizes to your configuration):

```bash
python -m torch.distributed.launch --nproc_per_node=8 ./train.py --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=2 --valid_batch_size=2
```
| Experiment           | NIST1  | NIST2  | NIST3  | NIST4  | BLEU1  | BLEU2  | BLEU3  | BLEU4  | METEOR | entropy1 | entropy2 | entropy3 | entropy4 | diversity1 | diversity2 | avg_len |
|----------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|----------|----------|----------|------------|------------|---------|
| Human                | 2.4237 | 2.6244 | 2.6472 | 2.65   | 0.3408 | 0.1235 | 0.0572 | 0.0313 | 0.0831 | 6.5893   | 9.7423   | 10.4101  | 10.4450  | 0.1666     | 0.6701     | 18.7568 |
| Best System (Team B) | 2.3408 | 2.5102 | 2.522  | 2.523  | 0.4122 | 0.1435 | 0.0501 | 0.0183 | 0.0807 | 5.3832   | 7.6065   | 8.5304   | 9.0298   | 0.1089     | 0.3249     | 15.1327 |
| our system           | 2.5863 | 2.804  | 2.823  | 2.8246 | 0.3927 | 0.1416 | 0.0555 | 0.0231 | 0.0851 | 5.5791   | 8.5109   | 9.6872   | 10.0765  | 0.0913     | 0.3973     | 16.9484 |
| our system(bs)       | 2.5943 | 2.9163 | 2.9624 | 2.9681 | 0.4238 | 0.1918 | 0.1027 | 0.0605 | 0.0929 | 6.0815   | 8.7379   | 9.4037   | 9.5697   | 0.1573     | 0.5103     | 14.1603 |



## 6K multi-ref dataset

| Experiment                | NIST1  | NIST2  | NIST3  | NIST4  | BLEU1  | BLEU2  | BLEU3  | BLEU4  | METEOR   | entropy1 | entropy2 | entropy3 | entropy4 | diversity1 | diversity2 |
|---------------------------|--------|--------|--------|--------|--------|--------|--------|--------|----------|----------|----------|----------|----------|------------|------------|
| Human          | 2.9939 | 3.412  | 3.491  | 3.5033 | 0.3961 | 0.179  | 0.1071 | 0.0748 | 0.106361 | 6.864963 | 10.21325 | 10.97053 | 10.9951  | 0.145482   | 0.629633   |
| 117M from scratch  | 1.1628 | 1.2315 | 1.2394 | 1.2402 | 0.3469 | 0.0974 | 0.038  | 0.0177 | 0.061685 | 4.536772 | 6.142383 | 6.759354 | 7.110297 | 0.053304   | 0.159148   |
| 345M from scratch | 2.2252 | 2.5085 | 2.5535 | 2.5596 | 0.3523 | 0.1692 | 0.0834 | 0.0459 | 0.093443 | 5.198841 | 7.669613 | 8.623757 | 9.034887 | 0.066593   | 0.256432   |
| 117M finetuning    | 2.2483 | 2.3943 | 2.4058 | 2.4065 | 0.3543 | 0.1054 | 0.0385 | 0.0155 | 0.075305 | 5.14805  | 7.292704 | 8.118871 | 8.510193 | 0.080533   | 0.262462   |
| 345M finetuning   | 2.6727 | 3.0033 | 3.0523 | 3.0585 | 0.4097 | 0.1696 | 0.0831 | 0.0456 | 0.098122 | 5.276367 | 7.765976 | 8.727143 | 9.125835 | 0.06801    | 0.263088   |
| 762M from scratch  | 2.2361 | 2.5236 | 2.5709 | 2.5774 | 0.4253 | 0.1787 | 0.0907 | 0.0519 | 0.095325 | 5.347718 | 7.964725 | 8.944626 | 9.322642 | 0.074926   | 0.293058   |

<!--## ConvAI challenge -->

## Try our system
The live demo access is upon approval request. 

<!--This model should give a Hits@1 over 79, perplexity of 20.5 and F1 of 16.5 using the convai2 evaluation script (see below).

These numbers are slightly lower than the number we obtained in the ConvAI2 competition. Here is what you can tweak to reach the same results:

- in the ConvAI2 competition we also used tweaked position emebddings so that the history of the dialog always start at with the same embeddings. This is easy to add with pytorch-pretrained-bert and should improve the hits@1 metric.
- in the ConvAI2 competition we used a beam search decoder. While the results are better in term of f1 metric, our feeling is that the human experience is les compelling with beam search versus the nucleus sampling detector which is provided in the present repository.-->

<!--## Using the interaction script

The training script saves all the experiments and checkpoints in a sub-folder named with the timestamp of the experiment in the `./runs` folder of the repository base folder.

You can then use the interactive script to interact with the model simply by pointing to this folder.

Here is an example command line to run the interactive script:

```bash
python ./interact.py --model_checkpoint ./data/Apr17_13-31-38_thunder/  # run the interactive script with a training checkpoint
python ./interact.py  # run the interactive script with the finetuned model on our S3
```

The fine-tuned model will gives FINAL Hits@1: 0.715

The interactive script accept a few arguments to tweak the decoding algorithm:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset. If empty download from S3.
dataset_cache | `str` | `'./dataset_cache.bin'` | Path or url of the dataset cache
model | `str` | `"openai-gpt"` | Path, url or short name of the model
max_history | `int` | `2` | Number of previous utterances to keep in history
device | `str` | `cuda` if `torch.cuda.is_available()` else `cpu` | Device (cuda or cpu)
no_sample | action `store_true` | Set to use greedy decoding instead of sampling
max_length | `int` | `20` | Maximum length of the output utterances
min_length | `int` | `1` | Minimum length of the output utterances
seed | `int` | `42` | Seed
temperature | `int` | `0.7` | Sampling softmax temperature
top_k | `int` | `0` | Filter top-k tokens before sampling (`<=0`: no filtering)
top_p | `float` | `0.9` | Nucleus filtering (top-p) before sampling (`<=0.0`: no filtering)

## Running ConvAI2 evaluation scripts

To run the evaluation scripts of the ConvAI2 challenge, you first need to install `ParlAI` in the repo base folder like this:

```bash
git clone https://github.com/facebookresearch/ParlAI.git
cd ParlAI
python setup.py develop
```

You can then run the evaluation script from `ParlAI` base folder:

```bash
cd ParlAI
python ../convai_evaluation.py --eval_type hits@1  # to download and evaluate our fine-tuned model on hits@1 metric
python ../convai_evaluation.py --eval_type hits@1  --model_checkpoint ./data/Apr17_13-31-38_thunder/  # to evaluate a training checkpoint on hits@1 metric
```

The evaluation script accept a few arguments to select the evaluation metric and tweak the decoding algorithm:

Argument | Type | Default value | Description
---------|------|---------------|------------
eval_type | `str` | `"hits@1"` | Evaluate the model on `hits@1`, `ppl` or `f1` metric on the ConvAI2 validation dataset
model | `str` | `"openai-gpt"` | Path, url or short name of the model
max_history | `int` | `2` | Number of previous utterances to keep in history
device | `str` | `cuda` if `torch.cuda.is_available()` else `cpu` | Device (cuda or cpu)
no_sample | action `store_true` | Set to use greedy decoding instead of sampling
max_length | `int` | `20` | Maximum length of the output utterances
min_length | `int` | `1` | Minimum length of the output utterances
seed | `int` | `42` | Seed
temperature | `int` | `0.7` | Sampling softmax temperature
top_k | `int` | `0` | Filter top-k tokens before sampling (`<=0`: no filtering)
top_p | `float` | `0.9` | Nucleus filtering (top-p) before sampling (`<=0.0`: no filtering)

-->

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Citation

If you use this code in your research, you can cite our [arxiv paper]():

```bash
@article{
}
```
