###Here will be the main script for generating the models
import spacy
import nltk
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
#from transformers import RobertaModel
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import KFold

from tqdm import tqdm  # progress bar



from utils.NLP_utils import *
from utils.IOfunctions import *
from model.bilstm_crf  import BiLSTM_CRF

import time

torch.manual_seed(1)

### DUMP FOR TESTING delete


START_TAG = "<START>"
STOP_TAG = "<STOP>"

# EMBEDDING_DIM = 50
HIDDEN_DIM = 2


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
    "<PADDING>":31
}


# Make up some training data
# training_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]

def build_representation():

    # load our training data
    print("-----Reading and transforming training data-----")
    raw_data = read_raw_data('NER_TRAIN_JUDGEMENT.json')
    training_data = build_training_data(raw_data)

    print("-----Training data read and transformed-----")

    # build word to index dictionary
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    return training_data, word_to_ix


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


def gradient_descent(training_data, model, optimizer, word_to_ix):
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence, tags)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()

        #print(model.BiLSTM.weight)
        #assert(False)
        optimizer.step()


def calculate_loss(model, x, y):
    print("-----Calculating Validation Loss-----")
    loss = 0.0
    with torch.no_grad():
        for i in range(len(x)):
            loss += model.neg_log_likelihood(x[i], y[i]) 
    print(f"-----Validation Loss is {loss}-----")

    return loss


class POS_dataset(Dataset):
    def __init__(self, x, y):
        # Initialize data, download, etc.
        self.n_samples = len(x)

        # here the first column is the class label, the rest are the features
        self.x_data = x # size [n_samples, n_features]
        self.y_data = y # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



def build_lstm_model(epoch_count, batch_size, lr):

    # we will store the validation loss after every epoch
    validation_loss = []
    # we prepare input sequences
    training_data, word_to_ix = build_representation()

    #training_data = training_data[:100]
    # preparing glove word embedding
    filename = 'glove.6B.50d' 
    project_path = os.path.split(os.path.realpath(__file__))[0]
    datafile = os.path.join(project_path,'pretrained_models',f'{filename}.txt')

    print("-----Loading Glove embeddings-----")
    glove = read_WE(datafile)
    embedding_matrix = get_embedding_matrix(glove, word_to_ix)
    embedding_layer = create_emb_layer(torch.tensor(embedding_matrix))
    print("-----Glove embeddings loaded-----")


    # prepare model components   
    gans = BiLSTM_CRF(len(word_to_ix), ent_to_ix, embedding_layer, HIDDEN_DIM)
    optimizer = optim.SGD(gans.parameters(), lr=lr, weight_decay=1e-4)
    x  = []
    y  = []
    for sentence,targets in training_data:
        x.append(prepare_sequence(sentence, word_to_ix))
        y.append(torch.tensor([ent_to_ix[t] for t in targets], dtype=torch.long))    


    before_train = time.time()

    # prepare training batches
    time_elapsed = 0


    for epoch in range(1, epoch_count+1):
        print("---Starting epoch {}---".format(epoch))
        epoch_start = time.time()
        total_batches = len(training_data) // batch_size + 1 
        print(f'Total batches: {total_batches}')

        # initial batch
        training_batch = training_data[0 : batch_size]

        for j in range(total_batches):
            
            # update batch
            batch_start = j * batch_size
            batch_end = batch_start + batch_size
            training_batch = list(zip(x,y))[batch_start : batch_end]
            
            # training
            print(f"-----Starting batch num:{j}-----")
            gradient_descent(training_batch, model=gans, optimizer=optimizer, word_to_ix=word_to_ix)

            elapsed_train = time.time() - before_train
            print("-----Finished training in {}-----".format(elapsed_train))
            
        
        epoch_end = time.time() - epoch_start
        time_elapsed += epoch_end
        print("---Time elapsed after {}th epoch: {}---".format(epoch, round(epoch_end, 3)))
        print("TIME ELAPSED:", time_elapsed)
        validation_loss.append(calculate_loss(gans, x, y))
        
    # Check predictions after training
    return gans, validation_loss





"""
###Here will be the main script for generating the models
import spacy
sentences = ['hello can i have some pizza',
'do you want some tea']

nlp = nlp = spacy.load('en_core_web_trf')

# get two documents
doc_1 = nlp(sentences[0])
doc_2 = nlp(sentences[1])
# use the similarity method that is based on the vectors, on Doc, Span or Token
#print(doc_1.similarity(doc_2[0:7]))
print('------------doc1------------')
for tok in doc_1:
    print(tok.text, tok.pos_)
    print(doc_1._.trf_data.tensors)
    
"""

"""

    #####PADDING
    # Determine maximum length
    max_len = max([i.squeeze().numel() for i in x])
    # pad all tensors to have same length
    x = [torch.nn.functional.pad(i, pad=(0, max_len - i.numel()), mode='constant', value=0) for i in x]
    y = [torch.nn.functional.pad(i, pad=(0, max_len - i.numel()), mode='constant', value=0) for i in y]
    x = torch.stack(x)
    y = torch.stack(y)
    dataset = POS_dataset(x,y)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    print(f'Batch size {batch_size}')
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        #data = data.to(device=device).squeeze(1)
        #targets = targets.to(device=device)
        # forward
        #print(f'data {data} data.shape {torch.Size(data)}')
        scores = gans(data)
        loss = gans.neg_log_likelihood(data, targets)
        print(loss)
        validation_loss.append(loss)

#"""
