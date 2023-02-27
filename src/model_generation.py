###Here will be the main script for generating the models
import spacy
import nltk

###glove test added on 19.02. by yisheng###
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json


def test_glove(sentences):
    embeddings_dict = {}
    with open('src/glove/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embeddings_dict[word] = vector
    
    # print(embeddings_dict['the'])
    # print(len(embeddings_dict['the']))

    # test to trnasform a random sentence into glove embedding
    sentence_tokenized = []
    embedding = []
    
    for sentence in sentences:
        tokens = list(nltk.word_tokenize(sentence))
        sentence_tokenized.append(tokens)
        embedding.append([])
        print(tokens)
    for i in range(0, len(sentence_tokenized)):
        for j in range(0, len(sentence_tokenized[i])):
            embedding[i].append(embeddings_dict[sentence_tokenized[i][j]])
    
    print(embedding)



def test_roberta(sentences):
    # load one of the models listed at https://github.com/MartinoMensio/spacy-sentence-bert/
    #nlp = spacy_sentence_bert.load_model('roberta-base-nli-stsb-mean-tokens')
    nlp = nlp = spacy.load('en_core_web_trf')

    # get two documents
    doc_1 = nlp(sentences[0])
    doc_2 = nlp(sentences[1])
    # use the similarity method that is based on the vectors, on Doc, Span or Token
    #print(doc_1.similarity(doc_2[0:7]))
    print('------------doc1------------')
    for tok in doc_1:
        #print(tok.text, tok.pos_)
        # continute
        pass
       
    print('------------doc2------------')
    for tok in doc_2:
        print(tok.text, tok.pos_)
        print(tok)


"""
###Here will be the main script for generating the models
import spacy
sentences = ['hello can i have some pizza',
'do you want some tea']

nlp = nlp = spacy.load('en_core_web_trf')

# get two documents
doc_1 = nlp(sentences[0])
doc_2 = nlp(sentences[1])
# use the similarity method that is based on the vectors, on Doc, Span or Token
#print(doc_1.similarity(doc_2[0:7]))
print('------------doc1------------')
for tok in doc_1:
    print(tok.text, tok.pos_)
    print(doc_1._.trf_data.tensors)
    
"""

