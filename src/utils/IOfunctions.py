#Here should be defined the functions to read/write our:
# - Models
# - Pretransformed data

import json
import os
import numpy as np
from utils.NLP_utils import bio


def read_raw_data(filename):
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    project_path = os.path.split(cur_path)[0]
    datafile = os.path.join(project_path,'data',filename)
    with open(datafile, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def build_training_data(raw_data):
    
    training_data = []

    for i in range(0, len(raw_data)):
        training_data.append( bio(raw_data[i]))
        # print(training_data[i])
    
    return training_data


def read_glove_vector(glove):  
    
    embeddings_dict = {}
    with open(glove, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embeddings_dict[word] = vector
    
    return embeddings_dict


def get_embedding_matrix(embeddings_dict, word_to_ix):
    vocab_len = len(word_to_ix)
    embed_vector_len = embeddings_dict['the'].shape[0]

    # random matrix
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_len, embed_vector_len))

    # update weights of the tokens that exist in the embedding
    for word, index in word_to_ix.items():
        if embeddings_dict.get(word) is not None:
            embedding_matrix[index, :] = embeddings_dict.get(word)

    return embedding_matrix


def write_preprocessed_representation(name, data):
    pass


def read_preprocessed_representation():
    pass


def write_model():
    pass

def read_model():
    pass
