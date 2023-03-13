##Here we should define the specific NLP functions
##Basically anything that transforms text to some numerical representation
import torch

from nltk.tokenize import word_tokenize
import requests, zipfile, io
from clint.textui import progress

import os
#roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
#roberta.eval()  # disable dropout for evaluation

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0] + \
        torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def bio(data):
    tokenized = word_tokenize(data['data']['text'])
    tag = ['O' for i in range(0, len(tokenized))]
    tokenized_entities = []
    for entity in data['annotations'][0]['result']:
        text = word_tokenize(entity['value']['text'])
        labels = entity['value']['labels'][0]
        ent = {'text':text, 'labels':labels}
        tokenized_entities.append(ent)
    for i in range(0, len(tokenized)):
        for entity in tokenized_entities:
            
            # recognise the first token of the entity
            if tokenized[i] == entity['text'][0] and len(entity['text']) == 1:
                tag[i] = 'B-' + entity['labels']

            if tokenized[i] == entity['text'][0] and len(entity['text']) >= 2:
                # recognise the last token of the entity
                ending = len(entity['text'])-1
                if 0 < i+ending < len(tokenized) and tokenized[i+ending] == entity['text'][-1]:
                    tag[i] = 'B-' + entity['labels']
                    for j in range(1, len(entity['text'])):
                        tag[i+j] = 'I-' + entity['labels']
    
    return (tokenized, tag)

def download_pretrained_model(url, filename):
    #this function replaces(creating it anew if neede) 
    #the files for glove(zip + txt) 
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    project_path = os.path.split(cur_path)[0]
    datafile_zip = os.path.join(project_path,'pretrained_models',f'{filename}.zip')
    datafile = os.path.join(project_path,'pretrained_models')
    r = requests.get(url, stream=True)
    print(f'----- Downloading file form {url}(this can take a while)----')
    with open(datafile_zip, 'wb') as fd:
        total_length = int(r.headers.get('content-length'))
        chunk_size = 8192
        for chunk in progress.bar(r.iter_content(chunk_size=chunk_size),expected_size=(total_length/chunk_size) + 1):
            if chunk:
                fd.write(chunk)
                fd.flush()
    print('-----Extracting glove pretrained zip...-----')
    with zipfile.ZipFile(datafile_zip,'r') as z:
        z.extractall(datafile)
    