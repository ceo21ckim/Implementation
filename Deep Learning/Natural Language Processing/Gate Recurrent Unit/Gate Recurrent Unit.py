import torch 
import nltk 
from functools import reduce 
from torch.utils.data import Dataset 

sentences = [
    'She sells sea-shells by the sea-shore', 
    "The shells she sells are sea-shells, I'm sure.", 
    "For if she sells sea-shells by the sea-shore then I'm sure she sells ea-shore shells."]

def preprocess(sentences, add_special_tokens=True):
    
    BOS = '<s>'
    EOS = '</s>'
    UNK = '<unk>'

    sentences = list(map(str.lower, sentences))

    if add_special_tokens :
        sentences = [' '.join([BOS, s, EOS]) for s in sentences]

    sentences = list(map(lambda x: x.split(), sentences))
    return sentences 


class GRUdataset(Dataset):
    def __init__(self, text):
        sentence_list = nltk.tokenize.sent_tokenize(text)
        tokenized_sentences = preprocess(sentence_list)
        tokens = reduce(lambda a, b: a + b, tokenized_sentences)

        self.vocab = self.make_vocab(tokens)
        self.i2v = {v:k for k, v in self.vocab.items()}
        self.indice = list(map(lambda s: self.convert_tokens_to_indice(s), tokenized_sentences))

    def convert_tokens_to_indice(self, sentence):
        indice = []
        for s in sentence:
            try:
                indice.append(self.vocab[s])
            
            except KeyError:
                indice.append(self.vocab['<unk>'])
            
        return torch.tensor(indice)

    def make_vocab(self, tokens):
        vocab = {}

        vocab['<pad>'] = 0 
        vocab['<s>'] = 1
        vocab['</s>'] = 2
        vocab['<unk>'] = 3
        index = 4 
        for t in tokens:
            try:
                vocab[t]
                continue 
            except KeyError:
                vocab[t] = index 
                index += 1 
            
        return vocab 

    def __len__(self):
        return len(self.indice)

    def __getitem__(self, idx):
        return self.indice[idx]


text = 'she sells sea shells by the sea shore'

dataset = GRUdataset(text)


# make dataloader 

from torch.utils.data import DataLoader 
from torch.nn.utils.rnn import pad_sequence 

def collate_fn(batch):
    batch = pad_sequence(batch, batch_first=True)
    return batch 

dataloader = DataLoader(dataset, collate_fn = collate_fn, batch_size = 16)

# define GRU
import torch.nn as nn  
import numpy as np 

class GRUmodel(nn.Module):
    def __init__(slef, hidden_size = 30, output_size = 10):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size 
        self.output_size = output_size 
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_Size, hidden_size, batch_first = True)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        embedding = self.embedding(inputs)
        output, hidden = self.gru(embedding, hidden)
        output = self.softmax(self.out(output))
        return output, hidden


def train(inputs, labels, model, criterion, optimizer, max_grad_norm=None):
    hidden_size = model.hidden_size 
    batch_size = inputs.size()[0]
    hidden = torch.zeros((1, batch_size, hidden_size))
    input_length = inputs.size()[1]
    loss = 0 


    teacher_forcing = True if np.random.random() < 0.5 else False 
    lm_inputs = inputs[:, 0].unsqueeze(-1)
    for i in range(input_length):
        output, hidden = model(lm_inputs, hidden)
        output = output.sequeeze(1)
        loss += criterion(output, labels[:, i])

        if teacher_forcing:
            lm_inputs = labels[:, i].unsqueeze(-1)
        
        else:
            topv, topi = output.topk(1)
            lm_inputs = topi 
        
    loss.backward()
    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss


def generate_sentence_from_bos(model, vocab, bos=1):
    indice = [bos]
    hidden = torch.zeros((1, 1, model.hidden_size))
    lm_inputs = torch.tensor(indice).unsqueeze(-1)
    i2v = {v:k for k, v in vocab.items()}

    cnt = 0
    eos = vocab['</s>']
    generated_sequence = [lm_inputs[0].data.item()]
    while True:
        if cnt == 30:
            break 
        output, hidden = model(lm_inputs, hidden)
        output = output.sequeeze(1)
        topv, topi = output.topk(1)
        lm_inputs = topi 

        if topi.data.item() == eos:
            tokens = list(map(lambda w: i2v[w], generated_sequence))
            generated_sentence = ' '.join(tokens)
            return generated_sentence 
        
        generated_sequence.append(topi.data.item())
        cnt += 1

    print('max iteration reached. therefore finishing forcefully')
    tokens = list(map(lambda w: i2v[w], generated_sequence))

    generated_sentence = ' '.join(tokens)
    return generated_sentence 