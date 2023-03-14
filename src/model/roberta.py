# visualization libraries
import matplotlib.pyplot as plt
import numpy as np

# pytorch libraries
import torch # the main pytorch library
import torch.nn as nn # the sub-library containing Softmax, Module and other useful functions
import torch.optim as optim # the sub-library containing the common optimizers (SGD, Adam, etc.)

# huggingface's transformers library
from transformers import RobertaForTokenClassification, RobertaTokenizer

# huggingface's datasets library
import datasets

# the tqdm library used to show the iteration progress
import tqdm
#tqdmn = tqdm.notebook.tqdm
import pandas as pd
import sys
sys.path.insert(1, '/src/utils')

from utils.NLP_utils import *
from utils.IOfunctions import *


START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"

# entity to index dictionary
ent_to_ix = {
    "O": 0,
    START_TAG: 1,
    STOP_TAG: 2,
    
    "B-COURT": 3,
    "B-PETITIONER": 4,
    "B-RESPONDENT": 5,
    "B-JUDGE": 6,
    "B-LAWYER": 7,
    "B-DATE": 8,
    "B-ORG": 9,
    "B-GPE": 10,
    "B-STATUTE": 11,
    "B-PROVISION": 12,
    "B-PRECEDENT": 13,
    "B-CASE_NUMBER": 14,
    "B-WITNESS": 15,
    "B-OTHER_PERSON": 16,
    
    "I-COURT": 17,
    "I-PETITIONER": 18,
    "I-RESPONDENT": 19,
    "I-JUDGE": 20,
    "I-LAWYER": 21,
    "I-DATE": 22,
    "I-ORG": 23,
    "I-GPE": 24,
    "I-STATUTE": 25,
    "I-PROVISION": 26,
    "I-PRECEDENT": 27,
    "I-CASE_NUMBER": 28,
    "I-WITNESS": 29,
    "I-OTHER_PERSON": 30,
    PAD:31
}
ix_to_ent = {}
for ent in ent_to_ix:
    ix_to_ent[ent_to_ix[ent]] = ent


roberta_version = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(roberta_version)

def to_encoding(row):
    #print('*********************')
    #print(row)
    #turn tokens into Roberta input, pad, add attention mask
    encodings = tokenizer(row['sentence'], truncation=True, padding='max_length', is_split_into_words=True)
    #print('*****')
    # pad tags to max possible length
    #print('*****')
    labels = row['labels'] + [PAD] * (tokenizer.model_max_length - len(row['labels']))
    #print({ **encodings, 'labels': labels })
    #print('*****')
    return { **encodings, 'labels': labels }


def build_roberta_model(training_data):
    print(training_data[0])
    training_data = [{'sentence': i[0],'labels': i[1]} for i in training_data]
    training_data = {key: [d[key] for d in training_data] for key in training_data[0]}
    training_data = datasets.Dataset.from_dict(training_data)
    print('------------------------------')
    print(training_data[0])
    training_data = training_data.map(to_encoding)  
    print('------------------------------')
    print(training_data[0])
    # format the datasets so that we return only 'input_ids', 'attention_mask' and 'labels' 
    # making it easier to train and validate the model
    training_data.set_format(type='torch', columns=['sentence', 'attention_mask', 'labels'])

    # initialize the model and provide the 'num_labels' used to create the classification layer
    model = RobertaForTokenClassification.from_pretrained(roberta_version, num_labels=len(ent_to_ix))
    # assign the 'id2label' and 'label2id' model configs
    model.config.id2label = ix_to_ent
    model.config.label2id = ent_to_ix

    return prepared_data, model


def train_model(model):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")