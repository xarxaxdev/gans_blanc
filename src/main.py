import argparse
from model_generation import build_lstm_model
from utils.IOfunctions import *
from utils.NLP_utils import download_pretrained_model


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
        '--download_glove', dest='download_glove',
        help='Downloads the glove WE needed for our models.',
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
        print("Training BiLSTM-CRF with parameters {} epochs, {} batch size, and {} learning rate.".format(epoch_count,
                                                                                                        batch_size, lr))
        model,validation_loss = build_lstm_model(epoch_count=epoch_count, batch_size=batch_size, lr=lr)
        print(validation_loss)
        filename = f'bilstm_crf.e{epoch_count}.bs{batch_size}.lr{lr}'
        print('----- Saving model... -----')
        save_model(model,filename)
        print('----- Model saved. -----')



if __name__ == '__main__':
    main()


    #take argument value or default
    #preprocess_data = args.preprocess_data if args.preprocess_data else ''

    #if args.roberta_test:
        #print('-----testing roberta-----')
        #sentences = ['hello can i have some pizza',
        #'do you want some tea']

        #model_generation.test_roberta(sentences)

    #if args.glove_test:
        #print('-----testing glove-----')
        #sentences = ['hello can i have some pizza',
        #'do you want some tea']

        #model_generation.test_glove(sentences)

    #if args.bilstm_crf != '':
        #print('-----generating bilstm_crf-----')
        #model = build_lstm_model(epoch_count=epoch_count, batch_size=batch_size, lr=lr) # epoch, batch size

