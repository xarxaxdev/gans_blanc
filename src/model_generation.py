###Here will be the main script for generating the models

import spacy_sentence_bert

def test_roberta(sentences):
    # load one of the models listed at https://github.com/MartinoMensio/spacy-sentence-bert/
    nlp = spacy_sentence_bert.load_model('en_roberta_large_nli_stsb_mean_tokens')
    # get two documents
    doc_1 = nlp(sentences[0])
    doc_2 = nlp(sentences[1])
    # use the similarity method that is based on the vectors, on Doc, Span or Token
    #print(doc_1.similarity(doc_2[0:7]))
    print('------------doc1------------')
    print(doc_1)
    print('------------doc2------------')
    print(doc_2)
