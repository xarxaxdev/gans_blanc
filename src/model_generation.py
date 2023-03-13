###Here will be the main script for generating the models
import spacy
import nltk
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
#from transformers import RobertaModel

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


def do_an_echo(training_data, model, optimizer, word_to_ix):
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

        #print(model.BiLSTM.weight)
        #assert(False)
        optimizer.step()



def build_lstm_model(epoch_count, batch_size, lr):
    training_data, word_to_ix = build_representation()

    # preparing glove word embedding
    filename = 'glove.6B.50d' 
    project_path = os.path.split(os.path.realpath(__file__))[0]
    datafile = os.path.join(project_path,'pretrained_models',f'{filename}.txt')

    print("-----Loading Glove embeddings-----")
    glove = read_WE(datafile)
    embedding_matrix = get_embedding_matrix(glove, word_to_ix)
    embedding_layer = create_emb_layer(torch.tensor(embedding_matrix))
    print("----Glove embeddings loaded-----")

    gans = BiLSTM_CRF(len(word_to_ix), ent_to_ix, embedding_layer, HIDDEN_DIM)
    optimizer = optim.SGD(gans.parameters(), lr=lr, weight_decay=1e-4)

    # Check predictions before training
    #with torch.no_grad():
        #precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        #precheck_tags = torch.tensor([ent_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        #print(gans(precheck_sent))

    before_train = time.time()

    added_batch = 0
    # prepare training batches
    for j in range((len(training_data)-1) // batch_size + 1):

        batch_start = j * batch_size
        batch_end = batch_start + batch_size
        if batch_end < len(training_data): 
            training_batch = training_data[batch_start : batch_end]
        else:
            training_batch = training_data[batch_start : len(training_data)]

        # training
        print("-----Starting training-----")

        added_epoch = 0

        for epoch in range(1, epoch_count+1):
            print("---Starting epoch {}---".format(epoch))
            epoch_start = time.time()

            do_an_echo(training_batch, model=gans, optimizer=optimizer, word_to_ix=word_to_ix)
        
            for i in range(batch_start, batch_start+len(training_batch)):
                with torch.no_grad():
                    print('---training_data[' + str(i) + '][0]---')
                    # print(training_data[i][0])
                    precheck_sent = prepare_sequence(training_data[i][0], word_to_ix)
                    # print('-------precheck_sent--------')
                    # print(precheck_sent)
                    print('---y---')
                    print(torch.tensor([ent_to_ix[t] for t in training_data[i][1]], dtype=torch.long))
                    print('---yhat---')
                    print(gans(precheck_sent))

            epoch_end = time.time() - epoch_start
            added_epoch += epoch_end

            added_batch += epoch_end
            print("ADDED BATCH", added_batch)

            print("---Time elapsed after {}th epoch: {}---".format(epoch, round(added_epoch, 3)))

    after_train = time.time()
    elapsed = after_train-before_train
    print("-----Finished training at {}-----".format(elapsed))

    # Check predictions after training
    return gans





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

