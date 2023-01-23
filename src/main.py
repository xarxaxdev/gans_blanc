import argparse
import nltk

from nltk.tree import Tree
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

    args = parser.parse_args()

    #take argument value or default
    preprocess_data = arg.preprocess_data if arg.preprocess_data else '' 

    if args.preprocess_data != '':
        base_data = read_raw_data()        
        #NLPS UTILS WHICHEVER PREPROCESSING CALL FUNCTION
        our_preprocessed_data = base_data
        name_to_save = 'whatever the model and model parameters are to generate that data'
        write_preprocessed_representation(name_to_save,our_preprocessed_data)


if __name__ == "__main__":
    main()
