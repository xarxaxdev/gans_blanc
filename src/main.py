import argparse
import nltk

from nltk.tree import Tree

import model_generation
from utils.IOfunctions import read_raw_data

GRAMMAR_PATH = './data/atis-grammar-cnf.cfg'


def main():
    parser = argparse.ArgumentParser(
        description='Main Project for ENR'
    )

    parser.add_argument(
        '--preprocess_data', dest='preprocess_data',
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
    preprocess_data = args.preprocess_data if args.preprocess_data else ''

    if args.roberta_test != '':
        sentences = ['hello can i have some pizza',
        'do you want some tea']

        model_generation.test_roberta(sentences)

    if args.glove_test != '':
        sentences = ['hello can i have some pizza',
        'do you want some tea']
        
        read_raw_data('src/data/NER_TRAIN_JUDGEMENT.json')

        model_generation.test_glove(sentences)

    if args.preprocess_data != '':
        base_data = read_raw_data()        
        #NLPS UTILS WHICHEVER PREPROCESSING CALL FUNCTION
        our_preprocessed_data = base_data
        name_to_save = 'whatever the model and model parameters are to generate that data'
        write_preprocessed_representation(name_to_save,our_preprocessed_data)


if __name__ == "__main__":
    main()
