import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from utils.NLP_utils import *
from utils.IOfunctions import *
from model.bilstm_crf import BiLSTM_CRF
import time


torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
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
    "I-OTHER_PERSON": 30
}

def build_representation(dataset):

    # load our training data
    print("-----Reading and transforming data-----")
    raw_data = read_raw_data(dataset)
    training_data = build_training_data(raw_data)

    print("-----Data read and transformed-----")

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


def gradient_descent(training_data, model, optimizer):
    for sentence, tags in training_data:
        # clear gradients before each data instance
        model.zero_grad()

        # compute loss
        loss = model.neg_log_likelihood(sentence, tags)

        # run backward step
        loss.backward()

    optimizer.step()
    

def calculate_loss(model, x, y):
    print("-----Calculating validation loss-----")
    loss = 0.0
    with torch.no_grad():
        for i in range(len(x)):
            loss += model.neg_log_likelihood(x[i], y[i])
    loss = loss / len(x)

    print(f"-----Validation loss is {loss}-----")

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


def build_lstm_model(epoch_count, batch_size, lr, dataset):

    # store validation loss after every epoch
    validation_loss = []

    # prepare input sequences
    training_data, word_to_ix = build_representation(dataset)

    # preparing glove word embeddings
    filename = 'glove.6B.50d' 
    project_path = os.path.split(os.path.realpath(__file__))[0]
    datafile = os.path.join(project_path,'pretrained_models',f'{filename}.txt')

    print("-----Loading Glove embeddings-----")
    glove = read_WE(datafile)
    embedding_matrix = get_embedding_matrix(glove, word_to_ix)
    embedding_layer = create_emb_layer(torch.tensor(embedding_matrix))
    print("-----Glove embeddings loaded-----")

    # preparing model components
    bilstm_crf = BiLSTM_CRF(len(word_to_ix), ent_to_ix, embedding_layer, HIDDEN_DIM)
    optimizer = optim.SGD(bilstm_crf.parameters(), lr=lr, weight_decay=1e-4)
    x  = []
    y  = []
    for sentence,targets in training_data:
        x.append(prepare_sequence(sentence, word_to_ix))
        y.append(torch.tensor([ent_to_ix[t] for t in targets], dtype=torch.long))    

    before_train = time.time()
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
            print(f"-----Starting batch num:{j+1}-----")
            gradient_descent(training_batch, model=bilstm_crf, optimizer=optimizer)

            elapsed_train = time.time() - before_train
            print("-----Finished training in {}-----".format(elapsed_train))

        epoch_end = time.time() - epoch_start
        time_elapsed += epoch_end
        print("---Time elapsed after {}th epoch: {}---".format(epoch, round(epoch_end, 3)))

        # compute loss
        validation_loss.append(calculate_loss(bilstm_crf, x, y))
        
    # check predictions after training
    return bilstm_crf, validation_loss
