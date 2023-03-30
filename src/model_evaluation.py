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
    xs = [x for _,x in values ]
    ys = [y for y,_ in values ]
    x = max(xs)
    fig, ax = plt.subplots(figsize=(10, 4))
    # visualize the loss values
    ax.plot(xs,ys)
    # set the labels
    ax.set_ylabel(f'{y_name}')
    ax.set_xlabel(f'{x} {x_name}')
    fig.tight_layout()

    cur_path = os.path.split(os.path.realpath(__file__))[0]
    datafile = os.path.join(cur_path, 'plots', filename)
    print(f'Saving file in {datafile}')
    plt.savefig(f'{datafile}.png', bbox_inches='tight')



def flatten(l):
    return [item for sublist in l for item in sublist]

metrics = ['precision','recall','f1'] 

def compute_score(y_hat, y):
    y = [i.item() for i in flatten(y)]
    y_hat = [i.item() for i in flatten(y_hat)]
    labels = list(set(y))
    f1_metric = load('f1')
    precision_metric = load('precision')
    recall_metric = load('recall')

    scores = {}
    avg = 'macro'
    def add_metric(m_name,p,r):
        m = load(m_name)
        by_classes = m.compute(predictions=p, references= r, average=None)
        scores[f'{m_name}_by_class'] = by_classes[m_name].tolist()
        scores[f'{m_name}'] = m.compute(predictions=p, references=r, average=avg)[m_name]
    
    for i in metrics:
        add_metric(i,y_hat,y)

    scores['labels'] = labels
    return scores 

def write_scores(model_name, scores):
    text = ''
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    datafile = os.path.join(cur_path, 'evaluation_logs', model_name)

    #print(datafile)
    #assert(False)
    for m in metrics:
        text += f'{m}\t{scores[m]}\n'
    text += 'Metric'+'\t'
    for i in range(len(scores['labels'])):
        text += f'\t{scores["labels"][i]}'
    text += '\n'
    for m in metrics:
        text += m+'\t'
        for i in range(len(scores['labels'])):
            m_by_class = m +'_by_class'
            text += f'{scores[m_by_class][i]}\t'
        text += '\n'
    

    with open(datafile+'.csv', 'w') as f:
        f.write(text) 
    


def evaluate_model_bilstm_crf(model_path, dataset):    
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
    scores = compute_score(y_hat, y)
    scores['labels'] = [ix_to_ent[i] for i in scores['labels']]

    print('F1 score:', scores['f1'])
    print('Precision:', scores['precision'])
    print('Recall:', scores['recall'])
    print('F1_by_class score:', list(zip(scores['f1_by_class'],scores['labels'])))
    print('Precision_by_class score:', list(zip(scores['precision_by_class'],scores['labels'])))
    print('Recall_by_class score:', list(zip(scores['recall_by_class'],scores['labels'])))
    write_scores(model_path,scores)




def evaluate_model_roberta(model_path, dataset):
    print("-----Loading and preparing data...-----")
    # model initialization
    test_data, _  = read_raw_data(dataset)
    test_data = prepare_data(test_data,'testing')
    print("-----Loaded and prepared data-----")
    print("-----Loading model-----")
    model = load_model(model_path)
    model.eval()
    print("-----Model loaded-----")
    print("-----Running model through test data and scoring-----")    
    scores = predict_model(model,test_data)
    scores['labels'] = [ix_to_ent[i] for i in scores['labels']]

    print('F1 score:', scores['f1'])
    print('Precision:', scores['precision'])
    print('Recall:', scores['recall'])
    print('F1_by_class score:', list(zip(scores['f1_by_class'],scores['labels'])))
    print('Precision_by_class score:', list(zip(scores['precision_by_class'],scores['labels'])))
    print('Recall_by_class score:', list(zip(scores['recall_by_class'],scores['labels'])))
    print(scores)
    write_scores(model_path,scores)

    
