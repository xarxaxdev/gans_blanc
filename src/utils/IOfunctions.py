#Here should be defined the functions to read/write our:
# - Models
# - Pretransformed data

import json
from utils.NLP_utils import bio


def read_raw_data(path):
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in range(0, len(data)):
        bio(data[i])
    

def write_preprocessed_representation(name, data):
    pass


def read_preprocessed_representation():
    pass


def write_model():
    pass

def read_model():
    pass
