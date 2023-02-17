###Here will be the main script for generating the models
import spacy

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
        print(tok.text, tok.pos_)
    print('------------doc2------------')
    for tok in doc_2:
        print(tok.text, tok.pos_)
        print(tok)
