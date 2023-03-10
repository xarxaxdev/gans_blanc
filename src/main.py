import argparse
import nltk

from nltk.tree import Tree

import model_generation
from utils.IOfunctions import read_raw_data, build_training_data

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

        model_generation.test_glove(sentences)

    if args.preprocess_data != '':
        training_data,word_to_ix = build_representation()
        gans = BiLSTM_CRF(len(word_to_ix), 32, 2, 2, 0.25, ent_to_ix)
        optimizer = optim.SGD(gans.parameters(), lr=0.01, weight_decay=1e-4)
        

if __name__ == "__main__":
    main()
