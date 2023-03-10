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

import time

torch.manual_seed(1)

### DUMP FOR TESTING delete

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4


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
        return (torch.randn(4, 1, self.hidden_size // 2),
                torch.randn(4, 1, self.hidden_size // 2))

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
        terminal_var = forward_var + self.transitions[self.ent_to_ix[STOP_TAG]]
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
        lstm_out, self.hidden = self.BiLSTM(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_size)
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

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.entset_size), -10000.)
        init_vvars[0][self.ent_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.entset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.ent_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.ent_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    #def train(self, ):

        # model initialization:

        # optimizer initialization:

        #training phase





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
    return training_data,word_to_ix


def do_an_echo(training_data, model, optimizer,word_to_ix):
    print("---starting epoch {}---".format(epoch))
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        gans.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([ent_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = gans.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
    epoch_time = time.time()
    print("---time elapsed after {}th epoch: {}".format(epoch, epoch_time))



training_data,word_to_ix = build_representation()

gans = BiLSTM_CRF(len(word_to_ix), 32, 2, 2, 0.25, ent_to_ix)
optimizer = optim.SGD(gans.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
#with torch.no_grad():
    #precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    #precheck_tags = torch.tensor([ent_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    #print(gans(precheck_sent))

start = time.time()

print("-----starting training-----")

for epoch in range(3):
    do_an_echo(training_data, model= gans, optimizer = optimizer,word_to_ix=word_to_ix)

total = time.time()
print("-----finished training at {}-----".format(total))

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(gans(precheck_sent))


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

