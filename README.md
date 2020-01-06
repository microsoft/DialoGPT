# A State-of-the-Art Large-scale Pretrained Response generation model (DialoGPT)

This repository contains the source code and trained model for a large-scale pretrained dialogue response generation model. The [human evaluation results](#human_eval) indicate that the response generated from DialoGPT is comparable to human response quality under a single-turn conversation Turing test.

<!--See more details on our [project page](https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation/)-->

The repository is based on [huggingface pytorch-transformer](https://github.com/huggingface/transfer-learning-conv-ai) and [OpenAI GPT-2](https://github.com/openai/gpt-2), containing data extraction script, model training code and pretrained small (117M) medium (345M) and large (762M) model checkpoint.

The model is trained on 147M multi-turn dialogue from Reddit discussion thread. The largest model can be trained in several hours on a 8 V100 machines (however this is not required), with distributed training and FP16 option. 

The include script can be used to reproduce the results of DSTC-7 grounded dialogue generation challenge and a 6k multi-reference dataset created from Reddit data. 

Project webpage: [https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation/](https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation/)

ArXiv paper: [https://arxiv.org/abs/1911.00536](https://arxiv.org/abs/1911.00536)


## News ##
***(Update 01/06/2020) Some third-party decoding script implementations:***
- [https://colab.research.google.com/drive/1PslHE4Rl4RqSa20s7HEp0ZKITBir6ezE](https://colab.research.google.com/drive/1PslHE4Rl4RqSa20s7HEp0ZKITBir6ezE) A colab interactive notebook by qywu,[ref](https://github.com/microsoft/DialoGPT/issues/3#issuecomment-551410203)
- [https://github.com/andreamad8/DialoGPT2-Interact](https://github.com/andreamad8/DialoGPT2-Interact) An interactive script featuring multiturn chatbot by andreamad8,[ref](https://github.com/microsoft/DialoGPT/issues/3#issuecomment-551450016)
- [https://github.com/LHolten/DialoGTP-MMI-decoder](https://github.com/LHolten/DialoGTP-MMI-decoder) An MMI implementation by LHolten,[ref](https://github.com/microsoft/DialoGPT/issues/3#issuecomment-558318401)
- [https://colab.research.google.com/drive/1-_KjlAV3J1IVDw_9KogjKDCzgFY7Jp7E](https://colab.research.google.com/drive/1-_KjlAV3J1IVDw_9KogjKDCzgFY7Jp7E) A colab interactive notebook by illuminascent@Reddit,[ref](https://www.reddit.com/r/MachineLearning/comments/dt5woy/p_dialogpt_state_of_the_art_conversational_model/?st=k530k3oo&sh=f6cd20fd)


<!--**This github repository will be updated soon. Please stay tuned.**-->
## Minimal Computational Configurations
This code can be run on CPU, but it would be slow. We would recommend to use GPU to train and finetune all models. There is no minimal limit of the number of GPUs. However, if using distributed train for multiple GPUs configuration, the speed-up vs the number of GPUs is roughly sub-linear. To simulate the same batchsize when using less GPUs, please use a larger `gradient_accumulation_steps` in model training. 

The 117M and 345M model can be loaded in a single GPU with 12G memory. The 762M model would require a single GPU that has greater than 16G memory for efficient training. The training speed on a benchmark data with 50M training instances and V100 GPUs:

| n\_gpu           | epoch time (h) | token/sec  |
|----------------------|--------|--------|
| 1              | 118 | 10847 |
| 2              | 62 | 20645 |
| 4              | 34 | 37647 |
| 8              | 18 | 71356 |

Fine-tuning from our pretrained model on a new dataset typically requires 1-2 epochs.


## Setup & Installation (TL;DR)

We created a demo script `demo.py` to ease the difficulty of the deployment of this system. The `demo.py` contains a pipeline of **model downloading**, data extraction, data preprocessing and model training over a dummy dataset within one commandline. 



#### Train model with Conda Environment

Please use the below commandlines to clone, install the requirements and load the Conda environment (Note that Cuda 10 is required):


```bash
sudo apt-get install -y make wget gzip bzip2 xz-utils zstd
```

```bash
git clone https://github.com/microsoft/DialoGPT.git
cd DialoGPT
conda env create -f LSP-linux.yml -n LSP
conda activate LSP
```

If you run this on an architecture other than Linux, please use `LSP-generic.yml` instead of `LSP-linux.yml` but please note that the generic one is not tested in all platform, so the stablity can not be gauranteed.
To use fp16 training, please install apex by using commands below
  
```bash
conda activate LSP
git clone https://github.com/NVIDIA/apex
cd apex
git reset --hard 3d01e4a0a188cc8df54bc6e44cf5eb40ff6b4cc5
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
python3.6 demo.py
```

#### Train model with Docker environment
To start, first install the docker and Nvidia-docker from their official repos.
The image environment for running the code can be loaded as below:  

*Nvidia-docker v2.**

```bash
$ docker run --gpus all --ipc=host --rm -it -v $PWD:/workspace --network=host icaruszyz/large-scale-training:dialogpt bash
```
*Nvidia-docker v1.**

```bash
$ nvidia-docker --rm -it -v $PWD:/workspace --network=host icaruszyz/large-scale-training:dialogpt bash
```

Inside the docker container, run 

```bash
python demo.py
```



## Pipeline details

This section explains all components in the `demo.py`.

#### Data loading
Before running `demo.py`, you can set *DATA_FOLDER* (default value `./models`)  in `demo.py` as the place you want to download all the data and pretrained/fine-tuned models. Then simply run
```bash
python demo.py
```
to 

* automatically download models and data, 
* prepare raw data into db that is ready to use for the program,
* generate a training scripts.

Note that by default the `demo.py` will use a dummy data, please specify the Reddit training data by using option `--data`. Three options are  available:`dummy`,`small` and `full`. 

```bash
python demo.py --data small
python demo.py --data full
```

The small Reddit data is around 140MB and the full Reddit data is more than 27GB. You can prepare a cup of coffee when processing with the full Reddit data because **it takes a long time**!

#### Pretrained model

The pretrained and fine-tuned models are available on azure blobstorage.
Please run/see `demo.py` for more details about how to download/use those models. Or you could download directly by using the links in `demo_utils.py`.

#### Preparing data
First, use the `prepare4db.sh` to convert a tsv data file into the correct format that the following script can recognize.
The trainig data need to be then processed into a database file with below commandline:

```bash
python prepro.py --corpus $DATA_PATH
```



#### Using the training script

The training script can be used in single GPU or multiple GPU settings (distributed training across multiple GPUs within a single node):

```bash
python ./LSP_train.py  # Single GPU training
python -m torch.distributed.launch --nproc_per_node=8 ./LSP_train.py  # Training on 8 GPUs
```


The training script accept several arguments to tweak the training: 

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
no_token_id | `boolean` | `True` | If set True, using all-zeros token-type embedding.


During the training, two log files will be updated. The `train_log.txt` and `eval_log.txt` contains the model loss, perplexity and training speed (tokens/sec) statistics for the training and dev set. 

The log file and saved model checkpoint can be found in `./models/output_model`

#### Model decoding
We note that even with properly filtered Reddit dataset, sometimes our model can still generate moderately toxic/inappropriate responses. Due to this reason, we are unable to provide the decoding script at this time (The live demo and decoding script access is upon invitation only now ).
We are currently still working on a controlled decoding method to prevent this system from toxic generation. Please stay tuned. 

**See issues [#3](https://github.com/microsoft/DialoGPT/issues/3) and [Reddit discussions](https://www.reddit.com/r/MachineLearning/comments/dt5woy/p_dialogpt_state_of_the_art_conversational_model/) for some discussions on third-party decoding methods.** 

See below for some third-party decoding methods:
- [https://colab.research.google.com/drive/1PslHE4Rl4RqSa20s7HEp0ZKITBir6ezE](https://colab.research.google.com/drive/1PslHE4Rl4RqSa20s7HEp0ZKITBir6ezE) A colab interactive notebook by qywu,[ref](https://github.com/microsoft/DialoGPT/issues/3#issuecomment-551410203)
- [https://github.com/andreamad8/DialoGPT2-Interact](https://github.com/andreamad8/DialoGPT2-Interact) An interactive script featuring multiturn chatbot by andreamad8,[ref](https://github.com/microsoft/DialoGPT/issues/3#issuecomment-551450016)
- [https://github.com/LHolten/DialoGTP-MMI-decoder](https://github.com/LHolten/DialoGTP-MMI-decoder) An MMI implementation by LHolten,[ref](https://github.com/microsoft/DialoGPT/issues/3#issuecomment-558318401)
- [https://colab.research.google.com/drive/1-_KjlAV3J1IVDw_9KogjKDCzgFY7Jp7E](https://colab.research.google.com/drive/1-_KjlAV3J1IVDw_9KogjKDCzgFY7Jp7E) A colab interactive notebook by illuminascent@Reddit,[ref](https://www.reddit.com/r/MachineLearning/comments/dt5woy/p_dialogpt_state_of_the_art_conversational_model/?st=k530k3oo&sh=f6cd20fd)


## Models

We release 6 fine-tuned models which can be further fine-tuned on low-resource  user-customized dataset. The total parameters in these models range from 117M to 762M, in accord with OpenAI GPT-2 model sizes.   

| Model           |  Fine-tuned from GPT-2| Trained from scratch
|----------------------|--------|--------|
| DialoGPT 762M model| [link](https://convaisharables.blob.core.windows.net/lsp/multiref/large_ft.pkl) | [link](https://convaisharables.blob.core.windows.net/lsp/multiref/large_fs.pkl) |
| DialoGPT 345M model| [link](https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl) | [link](https://convaisharables.blob.core.windows.net/lsp/multiref/medium_fs.pkl) | 
| DialoGPT 117M model| [link](https://convaisharables.blob.core.windows.net/lsp/multiref/small_ft.pkl) | [link](https://convaisharables.blob.core.windows.net/lsp/multiref/small_fs.pkl) | 
| DialoGPT 345M model (reverse, for MMI)| [link](https://convaisharables.blob.core.windows.net/lsp/multiref/small_reverse.pkl) | -| 




The model files can be loaded exactly as the GPT-2 model checkpoint from Huggingface [pytorch-transformer](https://github.com/huggingface/transformers). Please download the required model configuration files (`merges.txt`, `config,json`, `vocab.json`) from `./configs/*`.

The reverse model is predicting the source from the target. This model is used  for MMI reranking. 

## Retraining full models

### Preparation

The first step to retrain the full models is to generate the aforementioned 27GB Reddit dataset. This involves downloading full Reddit submission and comments dumps from [https://files.pushshift.io/reddit](https://files.pushshift.io/reddit) and creating intermediate files, which overall require **700GB of local disk space**. Downloading and processing the full data requires about 1-2 days, depending on your (CPU) compute capabilties (e.g., ~24 hours with 8 cores on a recent computer). Assuming you ran the above setup and installation steps (conda activate LSP, etc.), you can create the full dataset by running either:

```
python demo.py --data full
```
or
```
cd reddit_extractor; SIZE=full make -j 8; cd ..
```

The former command calls the latter, so the two methods are equivalent. We recommend the former, as the latter is mostly useful if you run into any problem or want to customize any arguments (e.g., the `make` command lets you build only a subset of the data). Note that the downloading phase can be error prone, for example based on your geolocation (firewall, etc.). If the above commands fail to generate `data/train.tsv`, or if that file is not anywhere close to 27GB, it means something went wrong. In that case, you may want to inspect `reddit_extractor/wget-log` and `reddit_extractor/logs/*.log` for any obvious error (e.g., wget unable to download from pushshift.io). If error messages don't make sense to you, feel free to contact us. If so, please be sure to include any error messages gathered from these log files.

Training data statistics: the generated training tsv file should be roughly 26.8 GB uncompressed, with 146.8M training instances, 3.87B source tokens, and 2.14B target tokens (including utterance-level 0/1 weights).


### Training

We recommand generating the above data using the `demo.py --data full`, as it (1) generates the data, (2) converts it into DB format, and (3) trains a model using `python LSP_train.py`. Please directly edit `demo.py` if you want to customize any of the hyperparameters.


## Evaluations

#### DSTC-7 challenge

Our model achieved the state-of-the-art results in [DSTC-7 Challenge response generation task](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling). 


| Experiment         | NIST2 | NIST4 | BLEU2  | BLEU4 | METEOR | ENT-4 | DIST-1 | DIST-2 | Avg. Len |
|--------------------|-------|-------|--------|-------|--------|----------|------------|------------|---------|
| Human response     | 2.62  | 2.65  | 12.35% | 3.13% | 8.31%  | 10.45    | 16.66%     | 67.01%     | 18.8    |
| DSTC-7 Winner      | 2.51  | 2.52  | 14.35% | 1.83% | 8.07%  | 9.03     | 10.89%     | 32.49%     | 15.1    |
| DialoGPT 345M      | 2.80  | 2.82  | 14.16% | 2.31% | 8.51%  | **10.08**    | 9.13%      | 39.73%     | 16.9    |
| DialoGPT 345M (BS) | **2.92**  | **2.97**  | **19.18%** | **6.05%** | **9.29%**  | 9.57     | **15.73%**     | **51.03%**     | 14.2    |

where ENT represents the [Entropy score](https://arxiv.org/abs/1809.05972), and DIST represents the [Distinct score](https://arxiv.org/pdf/1510.03055.pdf). For all metrics except the average length, larger are better.  

<!--| Experiment           | NIST1  | NIST2  | NIST3  | NIST4  | BLEU1  | BLEU2  | BLEU3  | BLEU4  | METEOR | ENT-1 | ENT-2 | ENT-3 | ENT-4 | DIST-1 | DIST-2 | Len |
|----------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|----------|----------|----------|------------|------------|---------|
| Human                | 2.4237 | 2.6244 | 2.6472 | 2.65   | 0.3408 | 0.1235 | 0.0572 | 0.0313 | 0.0831 | 6.5893   | 9.7423   | 10.4101  | 10.4450  | 0.1666     | 0.6701     | 18.7568 |
| DSTC-7 Winner | 2.3408 | 2.5102 | 2.522  | 2.523  | 0.4122 | 0.1435 | 0.0501 | 0.0183 | 0.0807 | 5.3832   | 7.6065   | 8.5304   | 9.0298   | 0.1089     | 0.3249     | 15.1327 |
| DialoGPT           | 2.5863 | 2.804  | 2.823  | 2.8246 | 0.3927 | 0.1416 | 0.0555 | 0.0231 | 0.0851 | 5.5791   | 8.5109   | 9.6872   | 10.0765  | 0.0913     | 0.3973     | 16.9484 |
| DialoGPT(beam search)       | **2.5943**| **2.9163** | **2.9624** | **2.9681**| **0.4238** | **0.1918** | **0.1027** | **0.0605** | **0.0929** | **6.0815**   | **8.7379**   | 9.4037   | 9.5697   | 0.1573     | 0.5103     | 14.1603 |-->

Note that the superior automatic evaluation comparing to human responses does not necessary imply that our model achieves human parity. Please check out our paper for more detailed analysis.


To fine-tune the `345M` DialoGPT model on the DSTC-7 challenge data on a server with 8 V100 GPUs, please run the following commandline (The DSTC data can be found at [DSTC-7 repo](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling)): 

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 train_LSP.py --init_checkpoint ./models/medium/medium_ft.pkl --train_input_file ./data/DSTC_train.db --eval_input_file ./data/DSTC_valid.tsv --model_name_or_path ./model/medium/ --learning_rate 1e-4  --train_batch_size 64 --eval_batch_size 64 --no_token_id
```

The trained model can be found at [DSTC medium model](https://convaisharables.blob.core.windows.net/lsp/DSTC/medium_ft.pkl)


#### Evaluation

1. Please **downloads** the following 3rd-party packages and save into the empty folder `3rdparty`:
	* [**mteval-v14c.pl**](https://goo.gl/YUFajQ) to compute [NIST](http://www.mt-archive.info/HLT-2002-Doddington.pdf). You may need to install the following [perl](https://www.perl.org/get.html) modules (e.g. by `cpan install`): XML:Twig, Sort:Naturally and String:Util.
	* [**meteor-1.5**](http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz) to compute [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/index.html). It requires [Java](https://www.java.com/en/download/help/download_options.xml).


2. Please follow the [DSTC-7 official repo](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction) to extract the data, and put `data-official-test/test.refs.txt` into `./dstc/data/` folder.

3. Run the extraction script below to produce the human response hypothesis file `human.resp.txt`:

	```bash
	python extract_human.py
	```

4. Finally, to reproduce the results of human hypothesis on DSTC dataset, please run following commands under the repo folder:

	```bash
	python batch_eval.py
	```

The evaluation results will be generated in the folder `./dstc/eval/`


## 6K multi-ref dataset result

### Automatic evaluation

We test on 6K multi-ref dataset from Reddit (this test data will be release soon). The results are summarized in below

| Experiment         | NIST2 | NIST4 | BLEU2  | BLEU4 | METEOR | ENT-4 | DIST-1 | DIST-2 | Avg. Len |
|--------------------|-------|-------|--------|-------|--------|----------|------------|------------|---------|
| Human response     | 3.41  | 4.25  | 17.90% | 7.48% | 10.64% | 11       | 14.50%     | 63.00%     | 13.1    |
| DialoGPT 117M      | 2.39  | 2.41  | 10.54% | 1.55% | 7.53%  | 10.78    | 8.60%      | 39.90%     | 12.8    |
| DialoGPT 345M      | 3     | 3.06  | 16.96% | 4.56% | 9.81%  | 9.13     | 6.80%      | 26.30%     | 12.2    |
| DialoGPT 762M      | 2.84  | 2.9   | 18.66% | 5.25% | 9.66%  | 9.72     | 7.76%      | 29.93%     | 11.2    |
| DialoGPT 345M (BS) | **3.4**  | **3.5**   | **21.76%** | **7.92%** | 10.74%  | 10.48     | **12.38%**     | **48.74%**    | 11.3    |
| DialoGPT 345M (w/MMI)| 3.28  | 3.33 | 15.68% | 3.94% | **11.23%**  | **11.25**     | 9.39%    | 45.55%   | 17.2    |

### <a name="human_eval"></a>Human evaluation 

We further conduct human evaluations (6K examples for each methods, each example is evaluated by 3 human judges). The results show a strong evidence that our generation quality is towards approaching the quality of real human responses, under this non-interactive Turing test:


*Relevance*: A and B, which one is more relevant to the source prompt.

| System A | A Wins (%) | Ties (%) | B Wins (%) | System B|
|--------------------|-------|-------|--------|-------|
|DialoGPT 345M|2671      (45%)   | 513         (9%) |   2816       (47%)| Human responses|
|DialoGPT 345M| 3281       (72%)|    394         (9%)  |  882         (19%)| [PersonalityChat](https://docs.microsoft.com/en-us/azure/cognitive-services/project-personality-chat/overview)|
|DialoGPT 345M w/ MMI| **2871**     (48%)|    522         (9%)  |  2607      (43%)| Human responses|

*Informativeness*: A and B, which one is more contentful and informative. 

| System A | A Wins (%) | Ties (%) | B Wins (%) | System B|
|--------------------|-------|-------|--------|-------|
|DialoGPT 345M| 2722       (45%) |  234         (4%) |  3044       (51%)| Human responses|
|DialoGPT 345M|3490       (77%) |   206         (5%)  |  861         (19%)| [PersonalityChat](https://docs.microsoft.com/en-us/azure/cognitive-services/project-personality-chat/overview)|
|DialoGPT 345M w/ MMI| **3011**       (50%)|    234        (4%)  |  2755       (46%)| Human responses|


*Human-Like*: A and B, which one do you think is more likely to be generated by Human.

| System A | A Wins (%) | Ties (%) | B Wins (%) | System B|
|--------------------|-------|-------|--------|-------|
|DialoGPT 345M|2716       (45%)  | 263         (4%)  | 3021       (50%)| Human responses|
|DialoGPT 345M|3462       (76%) |  196         (4%)  | 899         (20%)| [PersonalityChat](https://docs.microsoft.com/en-us/azure/cognitive-services/project-personality-chat/overview)|
|DialoGPT 345M w/ MMI| **2978**      (50%)|    241         (4%)  |  2781        (46%)| Human responses|


Please see full details in our [arxiv paper](https://arxiv.org/abs/1911.00536). 




<!--Relevance
System Wins      (%)         Ties        (%)         Losses   (%)
2 vs 1     2671       (0.45)    513         (0.09)    2816       (0.47)
2 vs 3     3281       (0.72)    394         (0.09)    882         (0.19)
2 vs 4     2379       (0.40)    527         (0.09)    3094       (0.52)
2 vs 5     3019       (0.50)    581         (0.10)    2400       (0.40)
2 vs 6     2726       (0.45)    576         (0.10)    2698       (0.45)
 
Informativeness
System Wins      (%)         Ties        (%)         Losses   (%)
2 vs 1     2722       (0.45)    234         (0.04)    3044       (0.51)
2 vs 3     3490       (0.77)    206         (0.05)    861         (0.19)
2 vs 4     2474       (0.41)    257         (0.04)    3269       (0.54)
2 vs 5     3230       (0.54)    362         (0.06)    2408       (0.40)
2 vs 6     2856       (0.48)    303         (0.05)    2841       (0.47)
 
Human-Like
System Wins      (%)         Ties        (%)         Losses   (%)
2 vs 1     2716       (0.45)    263         (0.04)    3021       (0.50)
2 vs 3     3462       (0.76)    196         (0.04)    899         (0.20)
2 vs 4     2478       (0.41)    289         (0.05)    3233       (0.54)
2 vs 5     3233       (0.54)    340         (0.06)    2427       (0.40)
2 vs 6     2847       (0.47)    321         (0.05)    2832       (0.47)
--> 


<!--| Experiment                   | NIST1 | NIST2 | NIST3 | NIST4 | BLEU1  | BLEU2  | BLEU3  | BLEU4 | METEOR | ENT-4 | DIST-1 | DIST-2 |
|------------------------------|-------|-------|-------|-------|--------|--------|--------|-------|--------|----------|------------|------------|
| Human response               | 2.99  | 3.41  | 3.83  | 4.25  | 39.61% | 17.90% | 10.71% | 7.48% | 10.64% | 11       | 14.50%     | 63.00%     |
| DialoGPT 117M      | 2.25  | 2.39  | 2.41  | 2.41  | 35.43% | 10.54% | 3.85%  | 1.55% | 7.53%  | 10.78    | 8.60%      | 39.90%     |
| DialoGPT 345M      |**2.67** | **3**     | **3.05**  | **3.06**  | 40.97% | 16.96% | 8.31%  | 4.56% | **9.81%**  | 9.13     | 6.80%      | 26.30%     |
| DialoGPT 762M      | 2.51  | 2.84  | 2.89  | 2.9   | **44.45%** | **18.66%** | **9.32%**  | **5.25%** | 9.66%  | **9.72**     | **7.76%**      | **29.93%**    |-->

<!--| DialoGPT 762M  (beam search)| 2.51  | 2.9   | 2.97  | 2.98  | 44.84% | 21.08% | 12.01% | 7.57% | 10.11% | 10.06    | 11.62%     | 44.07%     |-->
<!--## ConvAI challenge -->
<!--| DialoGPT 345M (beam search)  | 2.94  | 3.4   | 3.49  | 3.5   | 45.27% | 21.76% | 12.51% | 7.92% | 10.74% | 10.48    | 12.38%     | 48.74%     |-->
## Generated Examples
* Multi-turn generation examples from an interactive environment:

|Role | Response |
|---------|--------|
|User | Does money buy happiness? |
| Bot |  Depends how much money you spend on it .|
|User | What is the best way to buy happiness ? |
| Bot | You just have to be a millionaire by your early 20s, then you can be happy . |
|User |This is so difficult ! |
| Bot | You have no idea how hard it is to be a millionaire and happy . There is a reason the rich have a lot of money |

* Multi-turn  self-playing  dialogue with user prompt

|Role | Response |
|---------|--------|
|User | what is the meaning of life ?|
| Bot | The  meaning  is  to  be  with  your  family  and friends .|
|Bot | I’m going to guess :  It means that your parents and friends have loved you . That’s the meaning , I believe . |
| Bot | I think  the  meaning  is  to  make  sure  you’re alive when you die . So you’ll be with your family , friends , and yourself .|
|Bot |It’s not , but thanks for trying to help . |





* Commonsense question answering examples

|User | Bot |
|---------|--------|
|who is the first president of the United States | George Washington |
|what is the boiling point of water? | I think it's about 212 F. |
|which one is bigger, sun or moon? | The sun . |
|which animal has black and white stripes? | A black and white striped cat . |

* Handling context that does not appear in training examples (note that training examples are sampled from 2005 to 2017)

|User | Bot |
|---------|--------|
|The trading war between China and US is still happening . | I'm sure the Chinese government and the U . S . will be fine with that as long as they keep their arms on the table and the money flowing . |
|who won the world cup in 2018 ? | Germany , I believe . |
|Nvidia's Titan RTX is really good . | I have this card and i love it . No complaints . |
|Can Porsche beat Tesla with its new Taycan EV ? | I don't think so . The Tesla is more powerful than the porsche .

* Our bot's response to philosophy questions

|User | Bot |
|---------|--------|
|What is the meaning of a good life ? | I think it's that we're all connected to our past lives and the meaning of life is to live the way you want to and how you want to . |
|Which is more important -- being successful or being happy ? | Being successful is just endless . Being happy is the only thing that matters . |
|Is peace the only way to stop war ? |  Not if the world is not willing to work with you . Peace is the ultimate goal , but peace requires more than that . |
| How much freedom should people have ? |  The right amount of freedom is the freedom to do as you please , as long as you don't hurt people or property .|


* Finally  

|User | Bot |
|---------|--------|
|What do you think of the DialoGPT repo ? | It's pretty good , but it's very easy to find a bug . |

Please start a issue if you spot any :)



## Try our system
The live demo and decoding script access is upon invitation only now. Please stayed tuned for the full release. 


<!--The live demo and decoding script access is upon approval request. Please apply [here](https://docs.google.com/spreadsheets/d/1epTNfaqva1isVO_o9pbyhVLsnzDn58dGkcLB0OUVcqs/edit?usp=sharing)-->

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

## Related Project

* Microsoft ICECAPS: [https://github.com/microsoft/icecaps](https://github.com/microsoft/icecaps). 

	As an orthogonal repository of this project, 
	Microsoft Icecaps is an open-source toolkit (in tensorflow) for building neural conversational systems. Icecaps provides an array of tools from recent conversation modeling and general NLP literature within a flexible paradigm that enables complex multi-task learning setups. 

* Pretrained UniLM: [https://github.com/microsoft/unilm](https://github.com/microsoft/unilm)
* MT-DNN: [https://github.com/namisan/mt-dnn](https://github.com/namisan/mt-dnn)
* A chinese counterpart of DialoGPT by yangjianxin1. [https://github.com/yangjianxin1/GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat). We are glad to see that the MMI strategy that we used in DialoGPT has also improved the performance for this project as well!

## Contact

Please contact [DialoGPT@microsoft.com](mailto:DialoGPT@microsoft.com) if you have any questions/suggestions. However, the response will be sporadic. Please expect delay.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Disclaimer

This repository aims to facilitate research in large-scale pretraining for conversational data. This toolkit contains only part of the modeling machinery needed to actually produce a model weight file in a running dialog. On its own, this model provides only information about the weights of various text spans; in order for a researcher to actually use it, they will need to bring conversational data of their own and decode the response generation from the pretrained system. Microsoft is not responsible for any generation from the 3rd party utilization of the pretrained system. 



## Citation
If you use this code in your research, you can cite our [arxiv paper](https://arxiv.org/abs/1911.00536):
```bash
@misc{zhang2019dialogpt,
    title={DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation},
    author={Yizhe Zhang and Siqi Sun and Michel Galley and Yen-Chun Chen and Chris Brockett and Xiang Gao and Jianfeng Gao and Jingjing Liu and Bill Dolan},
    year={2019},
    eprint={1911.00536},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```



