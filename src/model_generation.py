###Here will be the main script for generating the models
import spacy
import nltk

###glove test added on 19.02. by yisheng### (mostly seem useless)
import numpy as np
# from scipy import spatial
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import json

#pytorch test by delfina, will need it whether we end up discarding spacy or not
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
#from transformers import RobertaModel

from utils.NLP_utils import *
from utils.IOfunctions import read_raw_data, build_training_data
from model.bilstm_crf  import BiLSTM_CRF

import time

torch.manual_seed(1)

### DUMP FOR TESTING delete


START_TAG = "<START>"
STOP_TAG = "<STOP>"

EMBEDDING_DIM = 5
HIDDEN_DIM = 2


# ent_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
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
    print(f'{time.time()}-----READING DATA AND TRANSFORMING IT-----')
    raw_data = read_raw_data('NER_TRAIN_JUDGEMENT.json')
    training_data = build_training_data(raw_data)

    print(f'{time.time()}-----READ DATA AND TRANSFORMED IT-----')

    # should use embedding instead?
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return training_data, word_to_ix

def do_an_echo(training_data, model, optimizer, word_to_ix):
    training_data = training_data[:10] #DELETE
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([ent_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

        print(model(sentence_in))


def build_lstm_model():
    training_data,word_to_ix = build_representation()

    #gans = BiLSTM_CRF(len(word_to_ix), 8, 2, 2, 0.25, ent_to_ix)
    gans = BiLSTM_CRF(len(word_to_ix), ent_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(gans.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    #with torch.no_grad():
        #precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        #precheck_tags = torch.tensor([ent_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        #print(gans(precheck_sent))

    start = time.time()

    print("-----starting training-----")

    for epoch in range(100):
        print("---starting epoch {}---".format(epoch))
        start = time.time()
        do_an_echo(training_data, model = gans, optimizer = optimizer,word_to_ix=word_to_ix)
        with torch.no_grad():
            precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
            print(gans(precheck_sent))
        #epoch_time = time.time()
        print("---time elapsed after {}th epoch: {}---".format(epoch, round(time.time() - start,3)))

    total = time.time()
    print("-----finished training at {}-----".format(total))

    # Check predictions after training
    return gans


def test_glove(sentences):
    embeddings_dict = {}
    with open('src/glove/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embeddings_dict[word] = vector
    
    # print(embeddings_dict['the'])
    # print(len(embeddings_dict['the']))

    # test to trnasform a random sentence into glove embedding
    sentence_tokenized = []
    embedding = []
    
    for sentence in sentences:
        tokens = list(nltk.word_tokenize(sentence))
        sentence_tokenized.append(tokens)
        embedding.append([])
        print(tokens)
    for i in range(0, len(sentence_tokenized)):
        for j in range(0, len(sentence_tokenized[i])):
            embedding[i].append(embeddings_dict[sentence_tokenized[i][j]])
    
    print(embedding)



def test_roberta(sentences):
    # load one of the models listed at https://github.com/MartinoMensio/spacy-sentence-bert/
    #nlp = spacy_sentence_bert.load_model('roberta-base-nli-stsb-mean-tokens')
    nlp = nlp = spacy.load('en_core_web_trf')

    # get two documents
    doc_1 = nlp(sentences[0])
    doc_2 = nlp(sentences[1])
    # use the similarity method that is based on the vectors, on Doc, Span or Token
    #print(doc_1.similarity(doc_2[0:7]))
    print('------------doc1------------')
    for tok in doc_1:
        #print(tok.text, tok.pos_)
        # continute
        pass
       
    print('------------doc2------------')
    for tok in doc_2:
        print(tok.text, tok.pos_)
        print(tok)


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

