###Here will be the main script for generating the models
import spacy
import nltk

###glove test added on 19.02. by yisheng###
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json

#pytorch test by delfina, will need it whether we end up discarding spacy or not
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
#from transformers import RobertaModel

from utils.NLP_utils import *

torch.manual_seed(1)

### DUMP FOR TESTING delete

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

ent_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

class BiLSTM_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim, dropout, ent_to_ix, batch_first=True):
        super(BiLSTM_CRF, self).__init__()
        self.input_size = input_size # number of expected features in the input x
        self.hidden_size = hidden_size #number of features in the hidden state h
        self.num_layers = num_layers # number of recurrent layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.ent_to_ix = ent_to_ix # define helper ent_to_ix
        self.entset_size = len(ent_to_ix) # idem

        self.word_embeds = nn.Embedding(input_size, embedding_dim)

         # maps output of lstm into tag space
        self.BiLSTM = nn.LSTM(embedding_dim, hidden_size//2, num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_size, self.entset_size)

        self.transitions = nn.Parameter(torch.randn(self.entset_size, self.entset_size))
        self.transitions.data[ent_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, ent_to_ix[STOP_TAG]] = -10000
        #self.CRF = CRF(len(ent_to_ix), batch_first=True)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.entset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.ent_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.entset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.entset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def forward(self, feats):
        # hs = [batch_size x hidden_size]
        # cs = [batch_size x hidden_size]
        hs_forward = torch.zeros(x.size(0), self.hidden_size)
        cs_forward = torch.zeros(x.size(0), self.hidden_size)
        hs_backward = torch.zeros(x.size(0), self.hidden_size)
        cs_backward = torch.zeros(x.size(0), self.hidden_size)

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.ent_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.ent_to_ix[STOP_TAG], tags[-1]]
        return score

    #def train(self, ):

        # model initialization:

        # optimizer initialization:

        #training phase

gans = BiLSTM_CRF(128, 64, 2, 2, 0.25, ent_to_ix)


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

