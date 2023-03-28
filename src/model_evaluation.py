import os
import torch
import random
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from model_generation import ent_to_ix as ent_to_ix_bilstm 
from model_generation import ix_to_ent as ix_to_ent_bilstm
from utils.IOfunctions import *
from utils.NLP_utils import *
from model_generation import *
from model.roberta import prepare_data,predict_model
from model.roberta import ix_to_ent as ix_to_ent_roberta
from tqdm import tqdm

from evaluate import load


def save_plot_train_loss(train_loss, filename):
    epochs = len(train_loss)

    fig, ax = plt.subplots(figsize=(10, 4))
    # visualize the loss values
    ax.plot(train_loss)
    # set the labels
    ax.set_ylabel('Loss')
    ax.set_xlabel(f'{epochs} Epochs')
    fig.tight_layout()

    cur_path = os.path.split(os.path.realpath(__file__))[0]
    datafile = os.path.join(cur_path, 'plots', filename)

    plt.savefig(f'{datafile}.png', bbox_inches='tight')


def save_plot(values, x_name,y_name, filename):
    x = len(values)

    fig, ax = plt.subplots(figsize=(10, 4))
    # visualize the loss values
    ax.plot(values)
    # set the labels
    ax.set_ylabel(f'{y_name}')
    ax.set_xlabel(f'{x} {x_name}')
    fig.tight_layout()

    cur_path = os.path.split(os.path.realpath(__file__))[0]
    datafile = os.path.join(cur_path, 'plots', filename)

    plt.savefig(f'{datafile}.png', bbox_inches='tight')



def flatten(l):
    return [item for sublist in l for item in sublist]


def compute_score(y_hat, y, avg):
    y = flatten(y)
    y_hat = flatten(y_hat)

    f1_metric = load('f1')
    precision_metric = load('precision')
    recall_metric = load('recall')
    
    f1 = f1_metric.compute(predictions=y_hat, references=y, average=avg)
    precision = precision_metric.compute(predictions=y_hat, references=y, average=avg)
    recall = recall_metric.compute(predictions=y_hat, references=y, average=avg)
    
    return f1, precision, recall




def evaluate_model_bilstm_crf(model_path, dataset):
    
    # # model initialization
    # if dataset == 'NER_DEV_JUDGEMENT.json':
    #     _, word_to_ix = build_representation('NER_TRAIN_JUDGEMENT.json')
    # if dataset == 'NER_DEV_PREAMBLE.json':
    #     _, word_to_ix = build_representation('NER_TRAIN_PREAMBLE.json')
    
    # # update test data to representation
    # raw_data = read_raw_data(dataset)
    # test_data = build_data_representation(raw_data)

    test_data, word_to_ix = read_raw_data(dataset)

    # randomly assign unknown words to word_to_ix
    for sentence, tags in test_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = random.randint(0, 100)
    
    # load model
    print("-----Loading model-----")
    model = load_model(model_path)
    model.eval()
    print("-----Model loaded-----")

    x = []
    y = []
    y_hat = []
    for sentence, targets in test_data:
        x.append(prepare_sequence(sentence, word_to_ix))
        y.append(torch.tensor([ent_to_ix[t] for t in targets], dtype=torch.long))    
    
    print("-----Running through test data-----")
    for i in range(len(x)):
        y_hat.append(torch.tensor(model(x[i])[1]))
    

    # computing scores
    f1, precision, recall = compute_score(y_hat, y, None)

    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)





def evaluate_model_roberta(model_path, dataset):
    print("-----Loading and preparing data...-----")
    # model initialization
    if dataset == 'NER_DEV_JUDGEMENT.json':
        _, word_to_ix = build_representation('NER_DEV_JUDGEMENT.json')
    if dataset == 'NER_DEV_PREAMBLE.json':
        _, word_to_ix = build_representation('NER_DEV_PREAMBLE.json')
    
    # update test data to representation
    raw_data = read_raw_data(dataset)
    test_data = build_data_representation(raw_data)
    test_data = prepare_data(test_data,'testing')

    print("-----Loaded and prepared data-----")
    print("-----Loading model-----")
    model = load_model(model_path)
    model.eval()
    print("-----Model loaded-----")
    print("-----Running model through test data and scoring-----")    
    labels,f1,precision,recall,f1_all = predict_model(model,test_data)
    labels = [ix_to_ent_roberta[i] for i in labels]
    
    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_by_class score:', list(zip(f1_all,labels)))
    
    
    
