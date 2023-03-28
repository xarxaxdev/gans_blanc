import os
import torch
import random
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from utils.IOfunctions import *
from utils.NLP_utils import *
from model_generation import *
from model.roberta import prepare_data,predict_model
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
    y = [i.item() for i in flatten(y)]
    y_hat = [i.item() for i in flatten(y_hat)]
    labels = list(set(y))
    f1_metric = load('f1')
    precision_metric = load('precision')
    recall_metric = load('recall')

    f1_classes = f1_metric.compute(predictions=y_hat, references= y, average=None)
    f1_classes = f1_classes['f1'].tolist()
    f1 = f1_metric.compute(predictions=y_hat, references=y, average='macro')
    precision = precision_metric.compute(predictions=y_hat, references=y, average=None)
    recall = recall_metric.compute(predictions=y_hat, references=y, average=None)
    
    return labels, f1['f1'], precision['precision'].tolist(), recall['recall'].tolist(), f1_classes




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
    labels, f1, precision, recall, f1_all = compute_score(y_hat, y, None)
    labels = [ix_to_ent[i] for i in labels]

    print('F1 score:', f1)
    print('Precision:', list(zip(precision,labels))
    print('Recall:', list(zip(recall,labels)))
    print('F1_by_class score:', list(zip(f1_all,labels)))





def evaluate_model_roberta(model_path, dataset):
    print("-----Loading and preparing data...-----")
    # model initialization
    #if dataset == 'NER_DEV_JUDGEMENT.json':
    #    _, word_to_ix = build_representation('NER_DEV_JUDGEMENT.json')
    #if dataset == 'NER_DEV_PREAMBLE.json':
    #    _, word_to_ix = build_representation('NER_DEV_PREAMBLE.json')
    
    # update test data to representation
    #raw_data = read_raw_data(dataset)
    #test_data = build_data_representation(raw_data)
    test_data, _  = read_raw_data(dataset)
    test_data = prepare_data(test_data,'testing')

    print("-----Loaded and prepared data-----")
    print("-----Loading model-----")
    model = load_model(model_path)
    model.eval()
    print("-----Model loaded-----")
    print("-----Running model through test data and scoring-----")    
    labels,f1,precision,recall,f1_all = predict_model(model,test_data)
    labels = [ix_to_ent[i] for i in labels]
    
    print('F1 score:', f1)
    print('Precision:', list(zip(precision,labels))
    print('Recall:', list(zip(recall,labels)))
    print('F1_by_class score:', list(zip(f1_all,labels)))
    
    
    
