import torch
import torch.nn as nn
from utils.NLP_utils import *


START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_CRF(nn.Module):
    def __init__(self, input_size, ent_to_ix, embedding, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.input_size = input_size
        self.word_embeds, num_embeddings, embedding_dim = embedding
        self.hidden_dim = hidden_dim
        self.ent_to_ix = ent_to_ix
        self.entset_size = len(ent_to_ix)
        self.word_embeds = nn.Embedding(input_size, embedding_dim)
        self.BiLSTM = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.entset_size)
        self.transitions = nn.Parameter(torch.randn(self.entset_size, self.entset_size))
        self.transitions.data[ent_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, ent_to_ix[STOP_TAG]] = -10000


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # forward algorithm to compute the CRF layer partition function
        init_alphas = torch.full((1, self.entset_size), -10000.)
        # START_TAG has all of the score
        init_alphas[0][self.ent_to_ix[START_TAG]] = 0.

        # wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # iterate through the sentence
        for feat in feats:
            alphas_t = []  # forward tensors at this timestep
            for next_tag in range(self.entset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.entset_size)
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # the ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # the forward variable for this tag is log-sum-exp of all the scores
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.ent_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.BiLSTM(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.ent_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.ent_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # initialize the viterbi variables in log space
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
                # we don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.ent_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # pop off the start tag (we don't want to return that to the caller)
        start = best_path.pop()
        assert start == self.ent_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # don't confuse this with _forward_alg above
        # get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # find the best path, given the features
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
