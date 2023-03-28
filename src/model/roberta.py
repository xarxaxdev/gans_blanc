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

<<<<<<< HEAD
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"


BATCH_SIZE_TRAIN_CONCURRENT=4
BATCH_SIZE_VALIDATE_CONCURRENT=10*BATCH_SIZE_TRAIN_CONCURRENT

set_seed(123)
roberta_version = 'distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(roberta_version,add_prefix_space=True)
PAD = tokenizer.pad_token
data_collator = DataCollatorForTokenClassification(tokenizer)

=======
roberta_version = 'distilroberta-base'
tokenizer = RobertaTokenizer.from_pretrained(roberta_version)
PAD = tokenizer.pad_token


>>>>>>> 47c3a50972bc2522386b0b9bb248f08fe724148c


entities = ['COURT','PETITIONER','RESPONDENT','JUDGE','LAWYER','DATE','ORG',
'GPE','STATUTE','PROVISION','PRECEDENT','CASE_NUMBER','WITNESS','OTHER_PERSON']

# entity to index dictionary
ent_to_ix = {'O':0, PAD:-100} #-100 is the ignore_index default 
ix_to_ent = {}
i = 1
for ent in entities:
    ent_to_ix[f'B-{ent}']= i
    i+=1
    ent_to_ix[f'I-{ent}'] = i 
    i+=1 
for ent in ent_to_ix:
    ix_to_ent[ent_to_ix[ent]] = ent

for k in sorted(ix_to_ent.keys()):
    print(f'{k}: {ix_to_ent[k]}')
<<<<<<< HEAD



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

=======

def to_encoding(row):
    #turn tokens into Roberta input, pad, add attention mask
    encodings = tokenizer(row['sentence'], truncation=True, padding='max_length', is_split_into_words=True)
    #row['sentence'] = row['sentence'] #+ [PAD] * (tokenizer.model_max_length - len(row['sentence']))
    # pad tags to max possible length
    #print(encodings.attention_mask)
    #encodings.attention_mask = ( encodings.attention_mask encodings.inpu
    #print(encodings.attention_mask)
    labels = row['labels'] + [tokenizer.pad_token] * (tokenizer.model_max_length - len(row['labels']))
    labels = [ ent_to_ix[i] for i in labels]
    labels = torch.from_numpy(np.asarray(labels))
    #print(labels)
    #print(encodings.input_ids)
    #print(len([x for x in labels if x!= -100]))
    #print(len([x for x in encodings.input_ids if x !=1 ]))
    #assert(False)
    #labels = torch.where(encodings.input_ids== labe.pad_token,labels,-100)
    return { **encodings, 'labels': labels }


def prepare_data(data,dataset_type):
    data = [{'sentence': i[0],'labels': i[1]} for i in data if len(i[0]) <= 512]
    data = {key: [d[key] for d in data] for key in data[0]}
    data = datasets.Dataset.from_dict(data)
    data = data.map(to_encoding,desc= f'Mapping {dataset_type} dataset')  
    # format the datasets so that we return only 'input_ids', 'attention_mask' and 'labels' 
    # making it easier to train and validate the model
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return data

def build_roberta_model_base(training_data,validation_data):
    # initialize the model and provide the 'num_labels' used to create the classification layer
    model = RobertaForTokenClassification.from_pretrained(roberta_version, num_labels=len(ent_to_ix))
    #if tokenizer.pad_token is None:
    #    tokenizer.add_special_tokens({'pad_token': PAD})
    #    model.resize_token_embeddings(len(tokenizer))
>>>>>>> 47c3a50972bc2522386b0b9bb248f08fe724148c
    # assign the 'id2label' and 'label2id' model configs
    model.config.id2label = ix_to_ent
    model.config.label2id = ent_to_ix

<<<<<<< HEAD
    training_data = training_data[:300]
    validation_data = validation_data[:100]
=======
    #training_data = training_data[:100]
    #validation_data = validation_data[:30]
>>>>>>> 47c3a50972bc2522386b0b9bb248f08fe724148c
    training_data = prepare_data(training_data,'training')
    validation_data = prepare_data(validation_data,'validation')
    
    return training_data, validation_data, model


<<<<<<< HEAD
=======
def compute_validation_loss(model,device, validation_data):
    model.eval()  # handle drop-out/batch norm layers
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE_VALIDATE_CONCURRENT)
    current_loss = 0
    curr_cases = 0
    with torch.no_grad():
        #for step,(x,y) in enumerate(validation_loader):
        #    out = model(**x)  # only forward pass - NO gradients!!
        #    loss += criterion(out, y)
        for i, batch in enumerate(tqdm(validation_loader,leave= False, desc="Validation progress:")):
            # move the batch tensors to the same device as the
            batch = { k:v.to(device) for k, v in batch.items() }
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            outputs = model(**batch)
            # the outputs are of shape (loss, logits)
            loss = outputs[0]
            current_loss += loss.item()
            curr_cases += BATCH_SIZE_VALIDATE_CONCURRENT
            
        
        # total loss - divide by number of batches
        val_loss = current_loss / len(validation_loader)
        return val_loss
>>>>>>> 47c3a50972bc2522386b0b9bb248f08fe724148c


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
    avg='macro'
    f1 = f1_metric.compute(predictions=y_hat, references= y, average=avg)
    precision = precision_metric.compute(predictions= y_hat, references= y, average=avg)
    recall = recall_metric.compute(predictions=y_hat, references= y, average=avg)
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
<<<<<<< HEAD
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
=======
    model.train().to(device)
    # initialize the Adam optimizer (used for training/updating the model)
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)
    train_data = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE_TRAIN_CONCURRENT, shuffle=True)
    training_loss = []
    validation_loss = []

    torch.cuda.empty_cache()
    gc.collect() 
    print('-----Beginning to train model...-----')

    # iterate through the data 'epochs' times
    for epoch in tqdm(range(epochs), leave = True, desc="Epoch progress:"):
        current_loss = 0
        curr_cases = 0
        # iterate through each batch of the train data
        for i, batch in enumerate(tqdm(train_data, leave = False, desc="Batch progress:")):
            # move the batch tensors to the same device as the
            batch = { k:v.to(device) for k, v in batch.items() }
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            outputs = model(**batch)
            # the outputs are of shape (loss, logits)
            loss = outputs[0]
            if i == 0 :
                print(batch['input_ids'][0])
            #loss_mask = tags
            #if i%10 == 0 and i> 0:
            #    length = batch['attention_mask'].sum(dim=1)[0]
            #    pred_values = torch.argmax(outputs[1], dim=2)[0][:length]
            #    print(pred_values)            
            # with the .backward method it calculates all 
            # of  the gradients used for autograd
            loss.backward()
            current_loss += loss.item()
            curr_cases += BATCH_SIZE_TRAIN_CONCURRENT  
            if curr_cases % batch_size == 0 and i > 0:#update every batch_size
                # update the model using the optimizer
                optimizer.step()
                # once we update the model we set the gradients to zero
                optimizer.zero_grad()
                # store the loss value for visualization
                training_loss.append(current_loss / curr_cases)
                current_loss = 0
                current_cases = 0
                torch.cuda.empty_cache() 
                gc.collect()
                #initially we thought about having more validation data points, 
                #but this took too long.
                #validation_loss.append(compute_validation_loss(model,device, val_data,batch_size))
                #must set model to training again, validation deactivates training
                #model.train().to(device)
        # update the model one last time for this epoch
        optimizer.step()
        optimizer.zero_grad()
        #Now we evaluate the model
        training_loss.append(current_loss / curr_cases)
        validation_loss.append(compute_validation_loss(model, device,val_data))
        #must set model to training again, validation deactivates training
        model.train().to(device)

>>>>>>> 47c3a50972bc2522386b0b9bb248f08fe724148c
    
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
    
    
    

