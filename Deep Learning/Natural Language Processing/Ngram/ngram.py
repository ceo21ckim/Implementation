
# lower
sentences = [
    'She sells sea-shells by the sea-shore', 
    "The shells she sells are sea-shells, I'm sure.", 
    "For if she sells sea-shells by the sea-shore then I'm sure she sells ea-shore shells."]

sentences = list(map(str.lower,sentences ))

# BOS/EOS token 
# BOS: Beginning Of Sentences 
# EOS: End Of Sentences

'''
아래의 경우는 n=2 즉, 2-gram인 경우의 예시.
3-gram 혹은 4-gram인 경우 첫 번째 토큰의 확률값을 계산하기 위해서는 
<s> <s> I 혹은 <s> <s> <s> I로 계산하여야 하기 때문에 BOSs를 생성
'''
BOS = '<s>'
EOS = '</s>'
n = 2 
BOSs = ' '.join([BOS]*(n-1) if n > 1 else [BOS]) 
sentences = [' '.join([BOSs, s, EOS]) for s in sentences]

sentences

# tokenizer 
from functools import reduce 
sentences = list(map(lambda x: x.split(' '), sentences))
tokens = reduce(lambda a, b: a+b, sentences) # 리스트를 unsqueeze()

# make <unk>
import nltk 
from functools import reduce 

UNK = '<unk>'
freq = nltk.FreqDist(tokens)
tokens = [t if freq[t] > 1 else UNK for t in tokens ]


# preprocessing
def preprocess(sentences, n):
    BOS = '<s>'
    EOS = '</s>'
    UNK = '<unk>'

    sentences = list(map(str.lower, sentences))

    BOSs = ' '.join([BOS]*(n-1) if n > 1 else [BOS])
    sentences = [' '.join([BOSs, s, EOS]) for s in sentences]

    sentences = list(map(lambda x: x.split(), sentences))
    tokens = reduce(lambda a, b: a + b, sentences)

    freq = nltk.FreqDist(tokens)
    tokens = [t if freq[t] > 1 else UNK for t in tokens]

    return tokens 


# solve zero-count 
# using Laplace smoothing 

bigram = nltk.ngrams(tokens, n=2)
vocab = nltk.FreqDist(bigram)

for k, v in vocab.items():
    a, b = k # if use 3-gram, then 'a, b, c' = k 
    print(f'{a},{b}: {v}')


# training n-gram 

import nltk 
a = ['a', 'b', 'b', 'b', 'a', 'a', 'a', 'c']
bigram = nltk.ngrams(a, n=2)
vocab = nltk.FreqDist(bigram)

def build_model(tokens, n):
    ngrams = nltk.ngrams(tokens, n)
    nvocab = nltk.FreqDist(ngrams)

    if n == 1:
        vocab = nltk.FreqDist(tokens)
        vocab_size = len(nvocab)
        return {v: c/vocab_size for v, c in vocab.items()}
    
    else:
        mgrams = nltk.ngrams(tokens, n-1)
        mvocab = nltk.FreqDist(mgrams)
        def ngram_prob(ngram, ncount):
            mgram = ngram[:-1]
            mcount = mvocab[mgram]
            return ncount / mcount 
        
        return {v: ngram_prob(v, c) for v, c in vocab.items()}