##Here we should define the specific NLP functions
##Basically anything that transforms text to some numerical representation


from nltk.tokenize import word_tokenize


def bio(data):
    tokenized = word_tokenize(data['data']['text'])
    tag = ['O' for i in range(0, len(tokenized))]

    # for entity in data['result']:
    #     word_tokenize(entity['value']['text'])
    #     for token in tokenized:
    #         pass
    
    for i in range(0, len(tokenized)):
        for entity in data['annotations'][0]['result']:
            entity_token = word_tokenize(entity['value']['text'])
            if tokenized[i] == entity_token[0]:
                tag[i] = 'B-' + entity['value']['labels'][0]
                if len(entity_token) >= 2:
                    for j in range(1, len(entity_token)):
                        tag[i+j] = 'I-' + entity['value']['labels'][0]

    print(data)
    print(tag)
    
    return tag

    # print(tokenized)
    # print(data['data']['text'])
