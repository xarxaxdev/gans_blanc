import os
import torch
import random
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from model_generation import ent_to_ix
from utils.IOfunctions import *
from utils.NLP_utils import *
from model_generation import *
from model.roberta import prepare_data,predict_model,ix_to_ent
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
    #plt.show()

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
    #plt.show()



def compute_f1(prediction, target):
    metric = MulticlassF1Score(num_classes=len(ent_to_ix), average='macro')
    return metric(prediction, target)

def compute_pre(prediction, target):
    metric = MulticlassF1Score(num_classes=len(ent_to_ix), average='macro')
    return metric(prediction, target)

def compute_rec(prediction, target):
    metric = MulticlassF1Score(num_classes=len(ent_to_ix), average='macro')
    return metric(prediction, target)


def evaluate_model(model_path, dataset):
    
    # model initialization
    if dataset == 'NER_DEV_JUDGEMENT.json':
        _, word_to_ix = build_representation('NER_TRAIN_JUDGEMENT.json')
    if dataset == 'NER_DEV_PREAMBLE.json':
        _, word_to_ix = build_representation('NER_TRAIN_PREAMBLE.json')
    
    dataset = 'NER_TRAIN_JUDGEMENT.json'
    
    # update test data to representation
    raw_data = read_raw_data(dataset)
    test_data = build_data_representation(raw_data)

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
        # print(i)
        # print(model(x[i]))
        y_hat.append(torch.tensor(model(x[i])[1]))
        
    prediction = torch.cat(y_hat)
    target = torch.cat(y)

    print("-----Computing scores-----")

    f1 = compute_f1(prediction, target)
    precision = compute_pre(prediction, target)
    recall = compute_rec(prediction, target)
    
    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)






def evaluate_model_roberta(model_path, dataset):
    
    # model initialization
    if dataset == 'NER_DEV_JUDGEMENT.json':
        _, word_to_ix = build_representation('NER_DEV_JUDGEMENT.json')
    if dataset == 'NER_DEV_PREAMBLE.json':
        _, word_to_ix = build_representation('NER_DEV_PREAMBLE.json')
    
    # update test data to representation
    raw_data = read_raw_data(dataset)
    print(len(raw_data))
    test_data = build_data_representation(raw_data)
    print(len(test_data))
    #test_data = start_stop_tagging(test_data)
    #test_data = test_data[:2]
    # randomly assign unknown words to word_to_ix
    #for sentence, tags in test_data:
        #for word in sentence:
            #if word not in word_to_ix:
                #word_to_ix[word] = random.randint(0, 100)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    # load model
    print("-----Loading model-----")
    model = load_model(model_path)
    model.eval().to(device)
    print("-----Model loaded-----")

    test_data = prepare_data(test_data,'testing')
    print("-----Running through test data-----")    
    labels,f1,precision,recall,f1_all = predict_model(model,test_data)
    labels = [ix_to_ent[i] for i in labels]
    
    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_by_class score:', list(zip(f1_all,labels)))
