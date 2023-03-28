import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from utils.NLP_utils import *
from utils.IOfunctions import *
from model.bilstm_crf import BiLSTM_CRF
import time
import random


HIDDEN_DIM=2

torch.manual_seed(1)
random.seed(10)

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



def build_representation(dataset):

    # load our training data
    print("-----Reading and transforming data-----")
    raw_data = read_raw_data(dataset)
    data = build_data_representation(raw_data)

    print("-----Data read and transformed-----")

    # build word to index dictionary
    word_to_ix = {}
    for sentence, tags in data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    return data, word_to_ix


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
    print("-----Calculating validaiton loss-----")
    loss = 0.0
    with torch.no_grad():
        for i in range(len(x)):
            loss += model.neg_log_likelihood(x[i], y[i])
    loss = loss / len(x)

    print(f"-----Validation loss: {loss}-----")

    return loss




def build_lstm_model(epoch_count, batch_size, lr, dataset):

    # store validation loss after every epoch
    validation_loss = []

    # prepare input sequences
    
    training_data, word_to_ix = build_representation(dataset)

    # split
    val_file = dataset.replace('.json','_VAL.json')
    tra_file = dataset.replace('.json','_TRA.json')

    # validation_data = read_raw_data(val_file)[0][0:10]
    # training_data = read_raw_data(tra_file)[0][0:10]
    
    validation_data = read_raw_data(val_file)[0]
    training_data = read_raw_data(tra_file)[0]

    # prepare validation set
    x_validation = []
    y_validation = []
    for sentence, targets in validation_data:
        x_validation.append(prepare_sequence(sentence, word_to_ix))
        y_validation.append(torch.tensor([ent_to_ix[t] for t in targets], dtype=torch.long))
    

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

    before_train = time.time()
    time_elapsed = 0


    for epoch in range(1, epoch_count+1):
        
        # shuffle and prepare training set
        random.shuffle(training_data)
        x_train = []
        y_train = []
        for sentence, targets in training_data:
            x_train.append(prepare_sequence(sentence, word_to_ix))
            y_train.append(torch.tensor([ent_to_ix[t] for t in targets], dtype=torch.long)) 

           
        print("---Starting epoch {}---".format(epoch))
        epoch_start = time.time()
        total_batches = len(training_data) // batch_size + 1 
        print(f'Total batches: {total_batches}')


        for j in range(total_batches):
            
            # update batch
            batch_start = j * batch_size
            batch_end = batch_start + batch_size
            training_batch = list(zip(x_train, y_train))[batch_start : batch_end]

            # training
            print(f"-----Starting batch num:{j+1}-----")
            gradient_descent(training_batch, model=bilstm_crf, optimizer=optimizer)

            elapsed_train = time.time() - before_train
            print("-----Finished training in {}-----".format(elapsed_train))

        epoch_end = time.time() - epoch_start
        time_elapsed += epoch_end
        print("---Time elapsed after {}th epoch: {}---".format(epoch, round(epoch_end, 3)))

        
        # check predictions after training
        validation_loss.append(calculate_loss(bilstm_crf, x_validation, y_validation))
        
        # for i in range(len(x_train)):
        #     # print(x_train[i])
        #     print('--------y--------')
        #     print(y_train[i])
        #     print('--------y_hat--------')
        #     print(bilstm_crf(x_train[i]))

    return bilstm_crf, validation_loss
