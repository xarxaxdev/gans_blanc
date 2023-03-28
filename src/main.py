import argparse
from model_generation import *
from model_evaluation import *
from utils.IOfunctions import *
from utils.NLP_utils import *
from model.roberta import *
import random

def params():
    parser = argparse.ArgumentParser(
        description='BiLSTM-CRF for ANLP Final Project'
    )

    parser.add_argument(
        '--bilstm_crf', dest='bilstm_crf',
        help='BiLSTM-CRF Model for Legal NER.',
        action='store_true'
    )

    parser.add_argument(
        '--roberta', dest='roberta',
        help='Roberta Model for Legal NER.',
        action='store_true'
    )
    
    parser.add_argument(
        '--roberta2', dest='roberta2',
        help='Roberta Model for Legal NER.',
        action='store_true'
    )
    
    parser.add_argument(
        '--download_glove', dest='download_glove',
        help='Downloads the glove WE needed for our models.',
        action='store_true'
    )

    parser.add_argument(
        '--split_datasets', dest='split_datasets',
        help='Generates the dataset splits for training and validation.',
        action='store_true'
    )


    parser.add_argument(
        '--evaluate_model', dest='evaluate_model',
        help='Evaluation of desired model with designated dataset.',
        action='store_true'
    )

    parser.add_argument(
        "--epochs", dest="epochs",
        help="number of epochs"
    )

    parser.add_argument(
        "--batch", dest="batch",
        help='batch size value'
    )

    parser.add_argument(
        "--lr", dest="lr",
        help='learning rate value'
    )

    parser.add_argument(
        "--model", dest="model",
        help='model to evaluate'
    )

    parser.add_argument(
        "--dataset", dest="dataset",
        help='dataset for training and evaluating model'
    )

    return parser.parse_args()

def main():
    args = params()
    if args.split_datasets:
        random.seed(123)
        files_to_save={}
        for dataset in ['NER_TRAIN_JUDGEMENT.json', 'NER_TRAIN_PREAMBLE.json']:
            data, word_to_ix = build_representation(dataset)
            testing_file = dataset.replace('.json', '_TES.json')  
            validation_file = dataset.replace('.json', '_VAL.json')  
            training_file = dataset.replace('.json', '_TRA.json')  
            random.shuffle(data)
            index1 = round(0.2 * len(data))
            index2 = round(0.4 * len(data))
            testing_data = data[:index1]
            validation_data = data[index1:index2]
            training_data = data[index2:]
            #now we build the vocabulary for BiLSTM
            #this code is a mess, should be done prettier
            vocabulary = set()
            for sentence,_ in validation_data + training_data:
                for w in sentence:
                    vocabulary.add(w)
            words = [i for i in word_to_ix.keys()]
            for w in words:
                if not(w in vocabulary):
                    del word_to_ix[w]
            word_to_ix2 = {}
            i = 0 
            for w in word_to_ix:
                word_to_ix2[w]= i
                i+=1
            word_to_ix = word_to_ix2
            #and we generate the new files to save
            files_to_save[testing_file] = (testing_data,word_to_ix)
            files_to_save[validation_file] = (validation_data,word_to_ix)
            files_to_save[training_file] = (training_data,word_to_ix)

        for f in files_to_save:
            save_raw_python(files_to_save[f],f)

    if args.download_glove:
        url_glove = 'https://nlp.stanford.edu/data/glove.6B.zip'
        filename = 'glove.6B' 
        download_pretrained_model(url=url_glove, filename = filename)
    
    
    if args.bilstm_crf:
        epoch_count = int(args.epochs)
        batch_size = int(args.batch)
        lr = float(args.lr)
        if args.dataset == 'judgement':
            dataset = 'NER_TRAIN_JUDGEMENT.json'
        if args.dataset == 'preamble':
            dataset = 'NER_TRAIN_PREAMBLE.json'

        print("Training BiLSTM-CRF with parameters {} epochs, {} batch size, and {} learning rate".format(epoch_count,
                                                                                                        batch_size, lr))
        model, validation_loss = build_lstm_model(epoch_count, batch_size, lr, dataset)
        print(validation_loss)
        filename = f'bilstm_crf.{args.dataset}.e{epoch_count}.bs{batch_size}.lr{lr}'
        print('----- Saving model... -----')
        save_model(model, filename)
        save_plot_train_loss(validation_loss, filename)
        print('----- Model saved. -----')
    
    
    if args.roberta:
        epochs = int(args.epochs)
        #batch_size = int(args.batch)
        lr = float(args.lr)
        if args.dataset == 'judgement':
            dataset = 'NER_TRAIN_JUDGEMENT.json'
        if args.dataset == 'preamble':
            dataset = 'NER_TRAIN_PREAMBLE.json'
        
        val_file = dataset.replace('.json','_VAL.json')  
        tra_file = dataset.replace('.json','_TRA.json')  
            
        (val_data, word_to_ix) = read_raw_python(val_file)
        (tra_data, word_to_ix) = read_raw_python(tra_file) 
 
        # get the untrained model
        tra_data, val_data, model = build_roberta_model_base(tra_data,val_data)
        # train it to our examples
        model,metrics = train_model(model, tra_data, val_data, epochs=epochs, lr=lr)
        filename = f'roberta.{args.dataset}.e{epochs}.lr{args.lr}'
        print(f'-----Saving model {filename}-----')
        save_model(model,filename)
        print('-----Model saved-----')
        print('-----Saving model metrics-----')
        for m in metrics:
            save_plot(metrics[m],'epochs',m, filename  +f'.{m}.training')
        print('-----Model training metrics-----')



    if args.evaluate_model:
        model = str(args.model)
        if args.dataset == 'judgement':
            dataset = 'NER_TRAIN_JUDGEMENT_TES.json'
        if args.dataset == 'preamble':
            dataset = 'NER_TRAIN_PREAMBLE_TES.json'
        if 'roberta' in model:
            evaluate_model_roberta(model,dataset)
        if 'bilstm_crf' in model:
            evaluate_model_bilstm_crf(model, dataset)


if __name__ == '__main__':
    main()
