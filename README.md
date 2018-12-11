# ActiveQA: Active Question Answering
This repo contains code for our paper [Ask the Right Questions: Active Question
 Reformulation with Reinforcement Learning](https://openreview.net/forum?id=S1CChZ-CZ).

Small forewarning, this is still much more of a research codebase than a
library. No support is provided.

*If you use this code for your research, please [cite the paper](#bibtex).*

## Introduction
ActiveQA is an agent that transforms questions online in order to find the best
answers. The agent consists of a Tensorflow model that reformulates questions
and an Answer Selection model. It interacts with an environment that contains
a question-answering system. The agent queries the environment with variants
of a question and calculates a score for the answer against the original
question. The model is trained end-to-end using reinforcement learning.

This version addresses the [SearchQA](https://arxiv.org/abs/1704.05179)
question-answering task, and the environment consists of the Bi-directional
Attention Flow ([BiDAF](https://github.com/allenai/bi-att-flow)) model of
[Seo et al. (2017)](https://openreview.net/forum?id=HJ0UKP9ge&noteId=HJ0UKP9ge).

## Setup
### Dependencies
We require tensorflow and many other supporting libraries. Tensorflow should be
installed separately following the docs. To install the other dependencies use

```
pip install -r requirements.txt
```

Note: We only ran this code with Python 2, so Python 3 is not officially
supported.

### Data
Download the source dataset from [SearchQA](https://github.com/nyu-dl/SearchQA),
[GloVe](https://nlp.stanford.edu/projects/glove/), and NLTK corpus and save
them in $HOME/data.

```
export DATA_DIR=$HOME/data
mkdir $DATA_DIR
```

#### Download
Download the SearchQA dataset (~600 MB) for training, testing, and validation
here: https://drive.google.com/open?id=1OxRhw81g7amW3aBd_iu2By5THysgr2uv

```
<Download the dataset to $DATA_DIR/SearchQA.zip>
unzip $DATA_DIR/SearchQA.zip -d $DATA_DIR
```

Download GloVe (~850 MB):

```
export GLOVE_DIR=$DATA_DIR/glove
mkdir $GLOVE_DIR

wget -c http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR
```

Download NLTK (for tokenizer). Make sure that nltk is installed!

```
python -m nltk.downloader -d $HOME/nltk_data punkt
```

Download the reformulator model pretrained on UN+Paralex datasets (~140 MB):

```
export PRETRAINED_DIR=$DATA_DIR/pretrained
mkdir $PRETRAINED_DIR

wget -c https://storage.googleapis.com/pretrained_models/translate.ckpt-1460356.zip -O $PRETRAINED_DIR/translate.ckpt-1460356.zip
unzip $PRETRAINED_DIR/translate.ckpt-1460356.zip -d $PRETRAINED_DIR
```

#### Preprocess
The SearchQA dataset requires a 2-step preprocessing:

1. Convert into SQuAD data format as the model was written to only work with
   that format.

   ```
   export SQUAD_DIR=$DATA_DIR/squad
   mkdir $SQUAD_DIR

   python -m searchqa.prepro \
   --searchqa_dir=$DATA_DIR/SearchQA \
   --squad_dir=$SQUAD_DIR
   ```

2. Preprocess the SearchQA dataset in SQuAD format (along with GloVe vectors)
   and save them in $PWD/data/squad (~60 minutes):

   ```
   python -m third_party.bi_att_flow.squad.prepro \
   --glove_dir=$GLOVE_DIR \
   --source_dir=$SQUAD_DIR
   ```

Note that Python2 and Python3 handle Unicode differently and hence the
preprocessing output differs. For converting the SearchQA format to SQuAD format
either version can be used; use Python3 for other datasets.

### gRPC
We need to compile the gRPC interface for the Environment Server.

```
chmod +x compile_protos.sh; ./compile_protos.sh
```

### Run Environment Server

The training requires running the environment [gRPC](https://grpc.io/)
server, which receives queries from the ActiveQA agent and sends back one
response per query.

   ```
   python -m px.environments.bidaf_server \
   --port=10000 \
   --squad_data_dir=data/squad \
   --bidaf_shared_file=data/bidaf/shared.json \
   --bidaf_model_dir=data/bidaf/
   ```

The checkpoint of a BiDAF model trained on SearchQA is already provided in
data/bidaf, so you don't have to train one yourself. However, if you want to
reproduce our training, clone the
[BiDAF repository](https://github.com/allenai/bi-att-flow) and run
```
python basic/cli.py \
--mode=trains \
--data_dir=data/squad \
--shared_path=data/bidaf/shared.json \
--init_lr=0.001 \
--num_steps=14000
```
### Reformulator Training

We first train reformulator from a model pretrained on UN and Paralex datasets.
It should take a week on a single P100 GPU to reach ~42 F1 score on SearchQA's
dev set.

```
export OUT_DIR=/tmp/active-qa
mkdir $OUT_DIR

export REFORMULATOR_DIR=$OUT_DIR/reformulator
mkdir $REFORMULATOR_DIR

echo "model_checkpoint_path: \"$PRETRAINED_DIR/translate.ckpt-1460356\"" > checkpoint
cp -f checkpoint $REFORMULATOR_DIR
cp -f checkpoint $REFORMULATOR_DIR/initial_checkpoint.txt

python -m px.nmt.reformulator_and_selector_training \
--environment_server_address=localhost:10000 \
--hparams_path=px/nmt/example_configs/reformulator.json \
--enable_reformulator_training=true \
--enable_selector_training=false \
--train_questions=$SQUAD_DIR/train-questions.txt \
--train_annotations=$SQUAD_DIR/train-annotation.txt \
--train_data=data/squad/data_train.json \
--dev_questions=$SQUAD_DIR/dev-questions.txt \
--dev_annotations=$SQUAD_DIR/dev-annotation.txt \
--dev_data=data/squad/data_dev.json \
--glove_path=$GLOVE_DIR/glove.6B.100d.txt \
--out_dir=$REFORMULATOR_DIR \
--tensorboard_dir=$OUT_DIR/tensorboard
```

Note: if you don't want to wait a week of training, you can download this
[checkpoint of the reformulator](https://storage.cloud.google.com/pretrained_models/translate.ckpt-6156696.zip)
trained on SearchQA, with dev set F1 score of 42.5. Note that this is not
the exact model analyzed in the paper, but one with equivalent performance.


### Selector Training

After training the reformulator, we can now train the selector. It should take
2-3 days on a single P100 GPU to reach ~47.5 F1 score on SearchQA's dev set.

```
python -m px.nmt.reformulator_and_selector_training \
--environment_server_address=localhost:10000 \
--hparams_path=px/nmt/example_configs/reformulator.json \
--enable_reformulator_training=false \
--enable_selector_training=true \
--train_questions=$SQUAD_DIR/train-questions.txt \
--train_annotations=$SQUAD_DIR/train-annotation.txt \
--train_data=data/squad/data_train.json \
--dev_questions=$SQUAD_DIR/dev-questions.txt \
--dev_annotations=$SQUAD_DIR/dev-annotation.txt \
--dev_data=data/squad/data_dev.json \
--glove_path=$GLOVE_DIR/glove.6B.100d.txt \
--batch_size_train=16 \
--batch_size_eval=64 \
--save_path=$OUT_DIR/selector \
--out_dir=$REFORMULATOR_DIR \
--tensorboard_dir=$OUT_DIR/tensorboard
```

Note: If you don't want to wait 2-3 days for the training to finish, you can
download a [checkpoint of the selector](https://storage.cloud.google.com/pretrained_models/selector.zip).
The checkpoint is trained on SearchQA, achieving an F1 score of ~47.5 on the dev
set.


## References

This repository relies on the work of the following repositories:

* [Neural Machine Translation (seq2seq)](https://github.com/tensorflow/nmt)
* [Bi-directional Attention Flow (BiDAF)](https://github.com/allenai/bi-att-flow)
* [SentencePiece](https://github.com/google/sentencepiece)

and uses data from the following sources:

* [GloVe](https://nlp.stanford.edu/projects/glove/)
* [SearchQA](https://github.com/nyu-dl/SearchQA)

# BibTex

```
@inproceedings{buck18,
  author    = {Christian Buck and
               Jannis Bulian and
               Massimiliano Ciaramita and
               Andrea Gesmundo and
               Neil Houlsby and
               Wojciech Gajewski and
               Wei Wang},
  title     = {Ask the Right Questions: Active Question Reformulation with Reinforcement
               Learning},
  booktitle = {Sixth International Conference on Learning Representations (ICLR)},
  year      = {2018},
  month     = {May},
  address   = {Vancouver, Canada},
  url       = {https://openreview.net/forum?id=S1CChZ-CZ},
}
```
