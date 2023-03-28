import numpy as np
# pytorch libraries
import torch # the main pytorch library


BATCH_SIZE_TRAIN_CONCURRENT=4
BATCH_SIZE_VALIDATE_CONCURRENT=12*BATCH_SIZE_TRAIN_CONCURRENT


# huggingface's transformers library
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer,set_seed,DataCollatorForTokenClassification,AutoTokenizer
# huggingface's datasets library
import datasets
from evaluate import load

# the tqdm library used to show the iteration progress
import sys

sys.path.insert(1, '/src/utils')
from utils.NLP_utils import *
from utils.IOfunctions import *

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"


BATCH_SIZE_TRAIN_CONCURRENT=4
BATCH_SIZE_VALIDATE_CONCURRENT=10*BATCH_SIZE_TRAIN_CONCURRENT

set_seed(123)
roberta_version = 'distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(roberta_version,add_prefix_space=True)
#PAD = tokenizer.pad_token
data_collator = DataCollatorForTokenClassification(tokenizer)







label_all_tokens = True
def prepare_data(data,dataset_type):
    data = [{'sentence': i[0],'labels': i[1]} for i in data if len(i[0]) <= 512]
    data = {key: [d[key] for d in data] for key in data[0]}
    tokenized_inputs = tokenizer(data["sentence"], truncation=True, padding='max_length',is_split_into_words=True)

    labels = []
    for i, label in enumerate(data[f"labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(PAD)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else PAD)
            previous_word_idx = word_idx

        label_ids = [ ent_to_ix[i] for i in label_ids]
        new_attention_mask = []
        #for l in label_ids:
        #    new_attention_mask.append( 1- (l in [0,-100]))
        #tokenized_inputs['attention_mask'][i] = new_attention_mask
        labels.append(torch.from_numpy(np.asarray(label_ids)))
    tokenized_inputs["labels"] = labels
    tokenized_inputs = datasets.Dataset.from_dict(tokenized_inputs)
    tokenized_inputs.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_inputs
 
    
    




def build_roberta_model_base(training_data,validation_data):
    # initialize the model and provide the 'num_labels' used to create the classification layer
    model = AutoModelForTokenClassification.from_pretrained(roberta_version, num_labels=len(ent_to_ix))

    # assign the 'id2label' and 'label2id' model configs
    model.config.id2label = ix_to_ent
    model.config.label2id = ent_to_ix

    training_data = training_data[:300]
    validation_data = validation_data[:100]
    training_data = prepare_data(training_data,'training')
    validation_data = prepare_data(validation_data,'validation')
    
    return training_data, validation_data, model




def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    y_hat = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    y = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    def flatten(l):
        return [item for sublist in l for item in sublist]
    y = flatten(y)
    y_hat = flatten(y_hat)
    y_classes = list(set(y))
    f1_metric = load('f1')
    precision_metric = load('precision')
    recall_metric = load('recall')
    f1_classes = f1_metric.compute(predictions=y_hat, references= y, average=None)
    f1_classes['f1']= f1_classes['f1'].tolist()

    f1 = f1_metric.compute(predictions=y_hat, references= y, average='macro')
    precision = precision_metric.compute(predictions= y_hat, references= y, average=None)
    precision['precision']= precision['precision'].tolist()
    recall = recall_metric.compute(predictions=y_hat, references= y, average=None)
    recall['recall']= recall['recall'].tolist()
    #print('F1 score:', f1)
    #print('Precision:', precision)
    #print('Recall:', recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_classes": f1_classes,
        "labels":y_classes
    }

def train_model(model,dataset,val_data,epochs = 3,lr = 1e-5):    
    print('-----Preparing for training-----')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set the model in 'train' mode and send it to the device
    filename = f'roberta.preamble.e{epochs}.lr{lr}'
    args = TrainingArguments(
        f"{filename}",save_strategy = "no",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE_TRAIN_CONCURRENT,
        per_device_eval_batch_size=BATCH_SIZE_VALIDATE_CONCURRENT,
        num_train_epochs=epochs,
        weight_decay=0.01#,
        #push_to_hub=True
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=dataset,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    tra_loss = []
    val_loss = []
    val_f1 = []
    print('-----Beginning to train model...-----')
    trainer.train()
    trainer.evaluate()
    
    for metrics in trainer.state.log_history:
        print(metrics)
        if 'eval_f1' in metrics:
            val_loss.append(metrics['eval_loss'])
            val_f1.append(metrics['eval_f1']['f1'])
        elif 'train_loss' in metrics :
            tra_loss.append(metrics['train_loss'])
        else :
            tra_loss.append(metrics['loss'])
    return model,{'val_loss': val_loss,'val_f1':val_f1,'tra_loss':tra_loss }
    
def predict_model(model,dataset):    
    print('-----Preparing for training-----')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set the model in 'train' mode and send it to the device
    filename = f'roberta.preamble'
    args = TrainingArguments(
        f"{filename}",save_strategy = "no",
        per_device_eval_batch_size=BATCH_SIZE_VALIDATE_CONCURRENT
    )
    
    trainer = Trainer(
        model,
        args,
        eval_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    metrics = trainer.evaluate()
    print(metrics)
    labels = metrics['eval_labels']
    f1 = metrics['eval_f1']['f1']
    precision = metrics['eval_precision']['precision']
    recall = metrics['eval_recall']['recall']
    f1_all = metrics['eval_f1_classes']['f1']
    
    return labels,f1,precision,recall,f1_all
    
    
    

