import os
import torch
import random
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from model_generation import ent_to_ix
from utils.IOfunctions import *
from utils.NLP_utils import *
from model_generation import *
from model.roberta import prepare_data
from tqdm import tqdm


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
        _, word_to_ix = build_representation('NER_TRAIN_JUDGEMENT.json')
    if dataset == 'NER_DEV_PREAMBLE.json':
        _, word_to_ix = build_representation('NER_TRAIN_PREAMBLE.json')
    
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

    #x = []
    #y = []
    #for sentence, targets in test_data:
    #    x.append(prepare_sequence(sentence, word_to_ix))
    #    y.append(torch.tensor([ent_to_ix[t] for t in targets], dtype=torch.long))    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = prepare_data(test_data,'testing')

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    print("-----Running through test data-----")    
    y_hat = []
    y = []
    for i, batch in enumerate(tqdm(test_loader,leave= False, desc="Testing progress:")):
        #print(test_data[i])
        with torch.no_grad():
            # move the batch tensors to the same device as the
            batch = { k:v.to(device) for k, v in batch.items() }
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            outputs = model(**batch)
            # the outputs are of shape (loss, logits)
        length = batch['attention_mask'].sum(dim=1)[0]
        #print('----outputs----')
        #print(outputs)
        #print('----')
        pred_values = torch.argmax(outputs[1], dim=2)[0][:length]
        #print('----pred_values----')
        #print(pred_values)
        #print('----')
        y_hat.append(pred_values)
        true_values = batch['labels'][0][:length]
        y.append(true_values)
    
    #for i in range(len(x)):
        #print(i)
        #print(model(x[i]))
        #y_hat.append(torch.tensor(model(x[i])[1]))
        
    prediction = torch.cat(y_hat)
    target = torch.cat(y)

    print("-----Computing scores-----")

    f1 = compute_f1(prediction, target)
    precision = compute_pre(prediction, target)
    recall = compute_rec(prediction, target)
    
    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)

