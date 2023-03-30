# BiLSTM-CRF and RoBERTa Models for Legal Named Entity Recognition
#### gans blank  ┐( ͡° ʖ̯ ͡°)┌ :swan:
## Project Description
This repository contains the final project for the course 'Advanced Natural Language Processing' of the M.Sc. Cognitive Systems: Language, Learning and Reasoning at Universität Potsdam.
This project deals with the [SemEval-2023 task 6: LegalEval , subtask B: Legal Entity Recognition (L-NER)](https://sites.google.com/view/legaleval/home#h.fbpoqsn0hjeh). You can find the paper presenting this task [here](https://aclanthology.org/2022.nllp-1.15/). This repository has been contributed by Guillem Gili i Bueno, Yi-Sheng Hsu and Delfina Jovanovich Trakál.

In this project, we propose two models for L-NER: a bidirectional long-short term memory neural network with a conditional random field layer (BiLSTM-CRF) and a RoBERTa model.
## Requirements
The packages required to run this project can be found in [requirements.txt](requirements.txt).
```bash
$ pip install -r requirements.txt
```
Make sure your Python version is compatible with PyTorch.

## Dataset
The data has been collected by the SemEval-2023 tasks 6 creators. It is divided into two categories, judgement and preamble, which don't present the same entity type and frequency. The .json files can be found under `src/data`.
More details on the data extraction and annotation processes can be found in the base paper linked above.

## Setup
### Training
#### BiLSTM + CRF
Download the pretrained Glove word embeddings:
```bash
$ python src/main.py --download_glove
```
Initialize either a BiLSTM-CRF or a RoBERTa model by using either the `--bilstm_crf` or the `--roberta` arguments. For training, specify for either model the number of epochs, the batch size, and the learning rate with the respective parameters `--epochs`,`--batch_size`, and `--lr`. Choose either the `judgement` or the `preamble` datasets with the argument `--dataset`. Here is an example:
```bash
$ python src/main.py --bilstm_crf --epochs 100 --batch_size 16 --lr 0.001 --dataset judgement
```
After successful training, the generated model will be saved to `src/generated_models`.
### Testing and Evaluation
Run `$ python src/main.py --evaluate_model` to test and evaluate either model on either judgement or preamble dev data. Specify which model to evaluate after the argument `--model` and on which dataset to test (`judgement` or `preamble`). We use F1 score. Here is an example:
```bash
$ python src/main.py --evaluate_model bilstm_crf.judgement.e100.bs512.lr0.001 --model judgement
```


## References
1. [Advanced: Making Dynamic Decisions and the Bi-LSTM CRF](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html) | PyTorch Tutorials
2. [F1-Score](https://huggingface.co/docs/evaluate/index) | Hugging Face evaluation Library
3. [pytorch-RoBERTa-named-entity-recognition](https://www.kaggle.com/code/eriknovak/pytorch-roberta-named-entity-recognition) | Kaggle
