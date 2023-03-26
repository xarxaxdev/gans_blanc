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
        val_split= 0.3
        for dataset in ['NER_TRAIN_JUDGEMENT.json', 'NER_TRAIN_PREAMBLE.json']:
            training_data, word_to_ix = build_representation(dataset)
            validation_file = dataset.replace('.json', '_VAL.json')  
            training_file = dataset.replace('.json', '_TRA.json')  
            random.shuffle(training_data)
            split_index = round(val_split * len(training_data))
            validation_data = (training_data[:split_index], word_to_ix)
            training_data = (training_data[split_index:], word_to_ix)
            save_raw_python(validation_data, validation_file)
            save_raw_python(training_data, training_file)        


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
        batch_size = int(args.batch)
        lr = float(args.lr)
        if args.dataset == 'judgement':
            dataset = 'NER_TRAIN_JUDGEMENT.json'
        if args.dataset == 'preamble':
            dataset = 'NER_TRAIN_PREAMBLE.json'
        
        val_file = dataset.replace('.json','_VAL.json')  
        tra_file = dataset.replace('.json','_TRA.json')  
            
        (val_data, word_to_ix) = read_raw_python(val_file)
        (tra_data, word_to_ix) = read_raw_python(tra_file) 
      
        val_data = start_stop_tagging(val_data)
        tra_data = start_stop_tagging(tra_data)
 
        # get the untrained model
        tra_data, val_data, model = build_roberta_model_base(tra_data,val_data)
        # train it to our examples
        model,tra_loss,val_loss = train_model(model, tra_data, val_data, epochs=epochs, batch_size = batch_size, lr=lr)
        filename = f'roberta.{args.dataset}.e{epochs}.bs{batch_size}.lr{lr}'
        print('-----Saving model-----')
        save_model(model,filename)
        print('-----Model saved-----')
        print('-----Saving model loss-----')
        save_plot_train_loss(tra_loss, filename + '.training')
        save_plot_train_loss(val_loss, filename + '.validation')
        print('-----Model training loss saved-----')


    if args.evaluate_model:
        model = str(args.model)
        if args.dataset == 'judgement':
            dataset = 'NER_DEV_JUDGEMENT.json'
        if args.dataset == 'preamble':
            dataset = 'NER_DEV_PREAMBLE.json'
        evaluate_model(model, dataset)


if __name__ == '__main__':
    main()
