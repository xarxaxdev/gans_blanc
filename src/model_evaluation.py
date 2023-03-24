import os
import torch
import random
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from model_generation import ent_to_ix
from utils.IOfunctions import *
from utils.NLP_utils import *
from model_generation import *


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
    metric = MulticlassF1Score(num_classes=len(ent_to_ix), average=None)
    return metric(prediction, target)

def compute_pre(prediction, target):
    metric = MulticlassPrecision(num_classes=len(ent_to_ix), average=None)
    return metric(prediction, target)

def compute_rec(prediction, target):
    metric = MulticlassRecall(num_classes=len(ent_to_ix), average=None)
    return metric(prediction, target)


def evaluate_model(model_path, dataset):
    
    # model initialisation
    if dataset == 'NER_DEV_JUDGEMENT.json':
        _, word_to_ix = build_representation('NER_TRAIN_JUDGEMENT.json')
    if dataset == 'NER_DEV_PREAMBLE.json':
        _, word_to_ix = build_representation('NER_TRAIN_PREAMBLE.json')
    
    # update test data to representation
    raw_data = read_raw_data(dataset)
    test_data = build_training_data(raw_data)

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


    batch = model_path.split(".")[2]
    batch_size = int("".join(list(filter(str.isdigit, batch))))
    total_batches = len(test_data) // batch_size + 1 
    
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
    print(f1)
    print(precision)
    print(recall)
    print(target[100:300])
    print(prediction[100:300])

