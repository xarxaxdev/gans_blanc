import argparse
import nltk

from nltk.tree import Tree

from model_generation import build_lstm_model
from utils.IOfunctions import read_raw_data, build_training_data

GRAMMAR_PATH = './data/atis-grammar-cnf.cfg'


def main():
    parser = argparse.ArgumentParser(
        description='Main Project for ENR'
    )

    parser.add_argument(
        '--bilstm_crf', dest='bilstm_crf',
        help='blablabal',
        action='store'
    )
    parser.add_argument(
        '--roberta_test', dest='roberta_test',
        help='blablabal',
        action='store'
    )
    parser.add_argument(
        '--glove_test', dest='glove_test',
        help='blablabal',
        action='store'
    )

    args = parser.parse_args()

    #take argument value or default
    #preprocess_data = args.preprocess_data if args.preprocess_data else ''

    if args.roberta_test:
        print('-----testing roberta-----')
        sentences = ['hello can i have some pizza',
        'do you want some tea']

        model_generation.test_roberta(sentences)

    if args.glove_test:
        print('-----testing glove-----')
        sentences = ['hello can i have some pizza',
        'do you want some tea']

        model_generation.test_glove(sentences)

    if args.bilstm_crf != '':        
        print('-----generating bilstm_crf-----')
        model = build_lstm_model(1, 10)

if __name__ == "__main__":
    main()
