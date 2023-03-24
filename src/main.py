import argparse
from model_generation import *
from model_evaluation import *
from utils.IOfunctions import *
from utils.NLP_utils import *
from model.roberta import *

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
        model, training_loss = build_lstm_model(epoch_count, batch_size, lr, dataset)
        print(training_loss)
        filename = f'bilstm_crf.{dataset}.e{epoch_count}.bs{batch_size}.lr{lr}'
        print('----- Saving model... -----')
        save_model(model, filename)
        save_plot_train_loss(training_loss, filename)
        print('----- Model saved. -----')
    
    
    if args.roberta:
        epochs = int(args.epochs)
        batch_size = int(args.batch)
        lr = float(args.lr)
        if args.dataset == 'judgement':
            dataset = 'NER_TRAIN_JUDGEMENT.json'
        if args.dataset == 'preamble':
            dataset = 'NER_TRAIN_PREAMBLE.json'
        
        training_data, word_to_ix = build_representation(dataset)
        # get the untrained model
        prepared_data, model = build_roberta_model_base(training_data)
        # train it to our examples
        model, training_loss = train_model(model, prepared_data, epochs=epochs, batch_size = batch_size, lr=lr)
        filename = f'roberta.e{epochs}.bs{batch_size}.lr{lr}'
        print('-----Saving model-----')
        save_model(model,filename)
        print('-----Model saved-----')
        print('-----Saving model training loss-----')
        save_plot_train_loss(training_loss, filename)
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
