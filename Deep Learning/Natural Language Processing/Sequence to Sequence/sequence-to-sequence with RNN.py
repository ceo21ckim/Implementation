# The most common sequence-to-sequence (Seq2Seq) model are encoder-decoder models.

import torch 
import torch.nn as nn 
import torch.optim as optim 

from torchtext.legacy.datasets import Multi30k 
from torchtext.legacy.data import Field, BucketIterator 

import spacy 
import numpy as np 
import random 
import math 
import time 

spacy_de = spacy.load('de_core_news_sm') # German
spacy_en = spacy.load('en_core_web_sm') # English

# introduces many short term dependencies in the data that make the optimization problem much easier. 
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

# torchtext 'Field' handle how data should be processed.

SRC = Field(
    tokenize = tokenize_de, 
    init_token = '<sos>', 
    eos_token = '<eos>', 
    lower = True)

TRG = Field(
    tokenize = tokenize_en, 
    init_token = '<sos>', 
    eos_token = '<eos>', 
    lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

# SRC include lookup Table
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

print(f'Unique tokens in source (de) vocabulary {SRC.vocab.__len__():,}')
print(f'Unique tokens in target (en) vocabulary {TRG.vocab.__len__():,}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device
)

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (h, c) = self.rnn(embedded)
        return h, c

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout_rate):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 
        self.n_layers = n_layers 
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout_rate)
        
        self.fc = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, h, c):

        input = input.unsqueeze(0) # input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        outputs, (h, c) = self.rnn(embedded, (h, c))
        
        predict = self.fc(outputs.squeeze(0))

        return predict, h, c 



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder 
        self.decoder = decoder 
        self.device = device 

        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimension of encoder and decoder must be equal!"
        
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and Decoder must have equal number of n_layers!"

        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim 

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            output[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[t] if teacher_force else top1 

        return outputs 


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256 
DEC_EMB_DIM = 256 
HID_DIM = 512
N_LAYERS = 2 
ENC_DROPOUT = 0.5 
DEC_DROPOUT = 0.5 

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'model has parameters {count_parameters(model):,} trainable parameters')


optimizer = optim.Adam(model.parameters(), lr = 1e-3)

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX).to(device)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.src.to(device)
        trg = batch.trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg).to(device)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    epoch_loss = 0
    for i, batch in enumerate(iterator):

        
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        output = model(src, trg).to(device)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)

        epoch_loss += loss 

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10 
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss 
        torch.save(model.state_dict(), 'Seq2Seq_with_RNN.pt')

    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')




model.load_state_dict(torch.load('tut1-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')