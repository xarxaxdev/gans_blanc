##Here we should define the specific NLP functions
##Basically anything that transforms text to some numerical representation


from nltk.tokenize import word_tokenize

#roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
#roberta.eval()  # disable dropout for evaluation

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0] + \
        torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))


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
            if tokenized[i] == entity_token[0] and len(entity_token) == 1:
                tag[i] = 'B-' + entity['value']['labels'][0]

            if tokenized[i] == entity_token[0] and len(entity_token) >= 2:
                # last token of the entity
                ending = len(entity_token)-1
                if 0 < i+ending < len(tokenized) and tokenized[i+ending] == entity_token[ending]:
                    tag[i] = 'B-' + entity['value']['labels'][0]
                    for j in range(1, len(entity_token)):
                        tag[i+j] = 'I-' + entity['value']['labels'][0]
            

    print(data)
    print(tag)
    
    return tag

    # print(tokenized)
    # print(data['data']['text'])
