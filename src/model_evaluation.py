import os
import torch
from torchmetrics.classification import MulticlassF1Score
from model_generation import ent_to_ix
from utils.IOfunctions import *
from utils.NLP_utils import *
from model_generation import *
from model.bilstm_crf import BiLSTM_CRF

# visualization libraries
import matplotlib.pyplot as plt
import numpy as np


### Here will be the main functions to evaluate the models
### Maybe draw some plots in src/plots




def save_plot_train_loss(train_loss,filename):
    epochs = len(train_loss)

    fig, ax = plt.subplots(figsize=(10, 4))
    # visualize the loss values
    ax.plot(train_loss)
    # set the labels
    ax.set_ylabel('Loss')
    ax.set_xlabel(f'{epochs} Epochs')
    fig.tight_layout()

    cur_path = os.path.split(os.path.realpath(__file__))[0]
    datafile = os.path.join(cur_path,'plots',filename)

    plt.savefig(f'{datafile}.png', bbox_inches='tight')
    #plt.show()


def compute_f1(prediction, target):
    metric = MulticlassF1Score(num_classes=len(ent_to_ix))
    return metric(prediction, target)


def evaluate_model(model_path, data_path):
    
    print("-----Initialsing model-----")
    # model initialisation
    training_data, word_to_ix = build_representation()
    
    # update word to ix
    raw_data = read_raw_data(data_path)
    test_data = build_training_data(raw_data)
    for sentence, tags in test_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    
    glove = read_WE('src/pretrained_models/glove.6B.50d.txt')
    embedding_matrix = get_embedding_matrix(glove, word_to_ix)
    embedding_layer = create_emb_layer(torch.tensor(embedding_matrix))

    bilstm_crf = BiLSTM_CRF(len(word_to_ix), ent_to_ix, embedding_layer, HIDDEN_DIM)
    bilstm_crf.load_state_dict(torch.load(model_path))
    model = bilstm_crf.eval()
    print("-----Model initialised-----")


    x  = []
    y  = []
    for sentence,targets in test_data:
        x.append(prepare_sequence(sentence, word_to_ix))
        y.append(torch.tensor([ent_to_ix[t] for t in targets], dtype=torch.long))    

    print(x[0])

    # Determine maximum length
    max_len = max([i.squeeze().numel() for i in x])
    # padding
    x = [torch.nn.functional.pad(i, pad=(0, max_len - i.numel()), mode='constant', value=0) for i in x]
    y = [torch.nn.functional.pad(i, pad=(0, max_len - i.numel()), mode='constant', value=0) for i in y]

    # print(x)
    # print(y)
    # x_total = torch.cat(x[0 : len(x)])
    # y_total = torch.cat(y[0 : len(y)])
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4)
    batch_size = 10
    total_batches = len(test_data) // batch_size + 1 
    print(f'Total batches: {total_batches}')


    # loss = 0
    # with torch.no_grad():
            
    #         for i in range(len(x)):
    #             loss += model.neg_log_likelihood(x[i], y[i])
    # print(loss)

    for j in range(total_batches):
        batch_start = j * batch_size
        batch_end = batch_start + batch_size
        # training_batch = list(zip(x,y))[batch_start : batch_end]
            
        # print(torch.cat(x[batch_start : batch_end]))
        # print(x[batch_start : batch_end])
        x_batch = torch.cat(x[batch_start : batch_end])
        y_batch = torch.cat(y[batch_start : batch_end])
        # test_batch = zip(x_batch, y_batch)[batch_start : batch_end]
        # test_batch = list(zip(x_batch, y_batch))[batch_start : batch_end]

        target = []
        prediction = []

        with torch.no_grad():
            

            for i in range(len(x_batch)):
                target.append(int(y_batch[i]))
                feats = model._get_lstm_features(x_batch[i])
                prediction.append(argmax(feats))
                # y_hat = ar
                # forward_score = model._forward_alg(feats)
                # print(forward_score)


            # print('-------test_data[0][0]--------')
            # precheck_sent = prepare_sequence(x, word_to_ix)
            # print('-------precheck_sent--------')
            # print(precheck_sent)
            # print('-------y--------')
            # print(torch.tensor([ent_to_ix[t] for t in test_data[0][1]], dtype=torch.long))
            # y = torch.tensor([ent_to_ix[t] for t in test_data[0][1]], dtype=torch.long)
            # print('-------yhat--------')
            # print(model(x))
            # print(x[i].shape)

            # y_hat = gradient_descent(test_batch, model, optimizer, word_to_ix)


            # y_hat = model(x[i])
            # print(y_hat)
            
        prediction = torch.tensor(prediction)
        target = torch.tensor(target)

        f1 = compute_f1(prediction, target)
        print(f1)