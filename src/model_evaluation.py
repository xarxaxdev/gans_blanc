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


# def compute_f1(prediction, target):
#     metric = MulticlassF1Score(num_classes=len(ent_to_ix), average='macro')
#     return metric(prediction, target)

# def compute_pre(prediction, target):
#     metric = MulticlassF1Score(num_classes=len(ent_to_ix), average='macro')
#     return metric(prediction, target)

# def compute_rec(prediction, target):
#     metric = MulticlassF1Score(num_classes=len(ent_to_ix), average='macro')
#     return metric(prediction, target)

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
    
    # model initialization
    if dataset == 'NER_DEV_JUDGEMENT.json':
        _, word_to_ix = build_representation('NER_TRAIN_JUDGEMENT.json')
    if dataset == 'NER_DEV_PREAMBLE.json':
        _, word_to_ix = build_representation('NER_TRAIN_PREAMBLE.json')
    
    # dataset = 'NER_TRAIN_JUDGEMENT.json'
    
    # update test data to representation
    raw_data = read_raw_data(dataset)
    test_data = build_data_representation(raw_data)

    print(len(test_data))
    
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
    

    # computing scores
    f1, precision, recall = compute_score(y_hat, y, 'macro')

    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)

    f1, precision, recall = compute_score(y_hat, y, None)

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

    test_data = build_data_representation(raw_data)
    #test_data = start_stop_tagging(test_data)
    #test_data = test_data[:10]
    # randomly assign unknown words to word_to_ix
    for sentence, tags in test_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = random.randint(0, 100)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    # load model
    print("-----Loading model-----")
    model = load_model(model_path)
    model.eval().to(device)
    print("-----Model loaded-----")

    test_data = prepare_data(test_data,'testing')

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    print("-----Running through test data-----")    
    y_hat = []
    y = []
    for i, batch in enumerate(tqdm(test_loader,leave= False, desc="Testing progress")):
        with torch.no_grad():
            # move the batch tensors to the same device as the
            batch = { k:v.to(device) for k, v in batch.items() }
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            logits = model(**batch).logits
            # the outputs are of shape (loss, logits)
        pred_values =  logits.argmax(-1)[0].tolist()
        true_values = batch['labels'][0].tolist()        
        length = true_values.index(-100)
        y_hat.append(pred_values[:length])
        y.append(true_values[:length])    


    # computing scores
    f1, precision, recall = compute_score(y_hat, y, 'macro')

    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)

    f1, precision, recall = compute_score(y_hat, y, None)

    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)