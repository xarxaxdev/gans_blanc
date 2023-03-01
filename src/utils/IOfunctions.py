#Here should be defined the functions to read/write our:
# - Models
# - Pretransformed data

import json
from utils.NLP_utils import tokenize, bio


def read_raw_data(path):
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def build_training_data(raw_data):
    
    training_data = []

    for i in range(0, len(raw_data)):
        training_data.append((tokenize(raw_data[i]), bio(raw_data[i])))
        # print(training_data[i])
    
    return training_data


def write_preprocessed_representation(name, data):
    pass


def read_preprocessed_representation():
    pass


def write_model():
    pass

def read_model():
    pass
