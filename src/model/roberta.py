# visualization libraries
import matplotlib.pyplot as plt
import numpy as np

# pytorch libraries
import torch # the main pytorch library
import torch.nn as nn # the sub-library containing Softmax, Module and other useful functions
import torch.optim as optim # the sub-library containing the common optimizers (SGD, Adam, etc.)
import os
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

# huggingface's transformers library
from transformers import RobertaForTokenClassification, RobertaTokenizer

# huggingface's datasets library
import datasets

# the tqdm library used to show the iteration progress
from tqdm import tqdm
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
    PAD:0,
    "O": 1,
    START_TAG: 2,
    STOP_TAG: 3,
    
    "B-COURT": 4,
    "B-PETITIONER": 5,
    "B-RESPONDENT": 6,
    "B-JUDGE": 7,
    "B-LAWYER": 8,
    "B-DATE": 9,
    "B-ORG": 10,
    "B-GPE": 11,
    "B-STATUTE": 12,
    "B-PROVISION": 13,
    "B-PRECEDENT": 14,
    "B-CASE_NUMBER": 15,
    "B-WITNESS": 16,
    "B-OTHER_PERSON": 17,
    
    "I-COURT": 18,
    "I-PETITIONER": 19,
    "I-RESPONDENT": 20,
    "I-JUDGE": 21,
    "I-LAWYER": 22,
    "I-DATE": 23,
    "I-ORG": 24,
    "I-GPE": 25,
    "I-STATUTE": 26,
    "I-PROVISION": 27,
    "I-PRECEDENT": 28,
    "I-CASE_NUMBER": 29,
    "I-WITNESS": 30,
    "I-OTHER_PERSON": 31,
}
ix_to_ent = {}
for ent in ent_to_ix:
    ix_to_ent[ent_to_ix[ent]] = ent


roberta_version = 'distilroberta-base'
tokenizer = RobertaTokenizer.from_pretrained(roberta_version)

def to_encoding(row):
    #turn tokens into Roberta input, pad, add attention mask
    encodings = tokenizer(row['sentence'], truncation=True, padding='max_length', is_split_into_words=True)
    row['sentence'] = row['sentence'] + [PAD] * (tokenizer.model_max_length - len(row['sentence']))
    # pad tags to max possible length
    labels = row['labels'] + [PAD] * (tokenizer.model_max_length - len(row['labels']))
    labels = [ ent_to_ix[i] for i in labels]
    labels = torch.from_numpy(np.asarray(labels))
    return { **encodings, 'labels': labels }


def build_roberta_model_base(training_data):
    training_data = [{'sentence': i[0],'labels': i[1]} for i in training_data]
    training_data = {key: [d[key] for d in training_data] for key in training_data[0]}
    training_data = datasets.Dataset.from_dict(training_data)
    training_data = training_data.map(to_encoding)  
    # format the datasets so that we return only 'input_ids', 'attention_mask' and 'labels' 
    # making it easier to train and validate the model
    training_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # initialize the model and provide the 'num_labels' used to create the classification layer
    model = RobertaForTokenClassification.from_pretrained(roberta_version, num_labels=len(ent_to_ix))
    # assign the 'id2label' and 'label2id' model configs
    model.config.id2label = ix_to_ent
    model.config.label2id = ent_to_ix

    return training_data, model


def train_model(model,dataset,epochs = 3,batch_size = 128,lr = 1e-5):    
    print('-----Preparing for training-----')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set the model in 'train' mode and send it to the device
    print(torch.cuda.is_available())
    print(device)

    model.train().to(device)
    # initialize the Adam optimizer (used for training/updating the model)
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)
    train_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    training_loss = []

    torch.cuda.empty_cache()
    gc.collect() 
    print('-----Beginning to train model...-----')

    # iterate through the data 'epochs' times
    for epoch in tqdm(range(epochs)):
        current_loss = 0
        curr_cases = 0
        # iterate through each batch of the train data
        for i, batch in enumerate(tqdm(train_data)):
            # move the batch tensors to the same device as the
            batch = { k:v.to(device) for k, v in batch.items() }
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            outputs = model(**batch)
            # the outputs are of shape (loss, logits)
            loss = outputs[0]
            # with the .backward method it calculates all 
            # of  the gradients used for autograd
            loss.backward()
            current_loss += loss.item()
            curr_cases += batch_size
            if i % 8 == 0 and i > 0:
                # update the model using the optimizer
                optimizer.step()
                # once we update the model we set the gradients to zero
                optimizer.zero_grad()
                # store the loss value for visualization
                training_loss.append(current_loss / curr_cases)
                current_loss = 0
                current_cases = 0
                torch.cuda.empty_cache() 
                gc.collect()
        training_loss.append(current_loss / curr_cases)
        # update the model one last time for this epoch
        optimizer.step()
        optimizer.zero_grad()
    
    return model,training_loss
