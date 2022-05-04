import functools 
import sys 
import datasets # huggingface 
import matplotlib.pyplot as plt 
import numpy as np
from pydantic import NumberNotMultipleError 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchtext 
import tqdm 

from torch.utils.data import DataLoader, Dataset 

SEED = 0 

np.random.seed(SEED)
torch.manual_seed(SEED)

train_data, test_data = datasets.load_dataset('imdb', split = ['train', 'test'])

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def tokenize_data(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    length = len(tokens)
    return {'tokens':tokens, 'length':length}

max_length = 256 

train_data = train_data.map(tokenize_data, fn_kwargs = {'tokenizer':tokenizer, 'max_length':max_length})
test_data = test_data.map(tokenize_data, fn_kwargs = {'tokenizer' :tokenizer, 'max_length':max_length})

test_size = 0.25 
train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data['train']
valid_data = train_valid_data['test']


min_freq = 5 
special_tokens = ['<unk>', '<pad>']

vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'], min_freq = min_freq, specials=special_tokens)

unk_index = vocab['<unk>']
pad_index = vocab['<pad>']

unk_index, pad_index 

vocab.set_default_index(unk_index)

def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids':ids}

train_data = train_data.map(numericalize_data, fn_kwargs = {'vocab':vocab})
valid_data = valid_data.map(numericalize_data, fn_kwargs = {'vocab':vocab})
test_data = test_data.map(numericalize_data, fn_kwargs = {'vocab':vocab})

train_data = train_data.with_format(type='torch', columns = ['ids', 'label', 'length'])
valid_data = valid_data.with_format(type='torch', columns = ['ids', 'label', 'length'])
test_data = test_data.with_format(type='torch', columns = ['ids', 'label', 'length'])

train_data[0]


######################### implementation ###########################

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, pad_index):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)

        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))

        else:
            hidden = self.dropout(hidden[-1])

        prediction = self.fc(hidden)
        return prediction


VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 300 
OUTPUT_DIM = len(train_data.unique('label'))
N_LAYERS = 2 
BIDIRECTIONAL = True 
DROPOUT_RATE = 0.5 


model = LSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT_RATE, pad_index)

# numel -> return the total number of elements in the input tensor
def count_parameters(model):
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

print(f'The model has {count_parameters(model):,} trainable parameters')


def initialize_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            
            elif 'weight' in name:
                nn.init.orthogonal_(param)

model.apply(initialize_weight)


vectors = torchtext.vocab.FastText()

pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
model.embedding.weight.data = pretrained_embedding

lr = 5e-4 
optimizer = optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset setting 
def collate(batch, pad_index):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = [i['length'] for i in batch]
    batch_length = torch.stack(batch_length)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {
        'ids':batch_ids, 
        'length':batch_length, 
        'label':batch_label
    }
    return batch 

batch_size = 512
collate = functools.partial(collate, pad_index=pad_index)


train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate)
test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate)


model = model.to(device)
criterion = criterion.to(device)

def get_accuracy(pred, label):
    batch_size, _ = pred.shape 
    predicted_classes = pred.argmax(dim=-1)
    # correct_pred = (predicted_classes == label).sum()
    correct_pred = predicted_classes.eq(label).sum()
    accuracy = correct_pred / batch_size 
    return accuracy 

def train(dataloader, model, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        ids = batch['ids'].to(device)
        length = batch['length']
        label = batch['label'].to(device)

        optimizer.zero_grad()
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        acc = get_accuracy(prediction, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)



def evaluate(dataloader, model, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            ids = batch['ids'].to(device)
            length = batch['length']
            label = batch['label'].to(device)

            prediction = model(ids, length)
            loss = criterion(prediction, label)
            acc = get_accuracy(prediction, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time 
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


NUM_EPOCHS = 20
best_valid_loss = float('inf')
import time

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    epoch_accs = []

    
    start_time = time.time()
    train_loss, train_acc = train(train_dataloader, model, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss 
        torch.save(model.state_dict(), 'LSTM.pt')

    print(f'Epoch : {epoch+1:02} | Epoch Time : {epoch_mins}m, {epoch_secs}s')
    print(f'\t Train Loss : {train_loss:.3f} | Train Acc : {train_acc*100:.2f}%')
    print(f'\t Val. Loss : {valid_loss:.3f}  | Val. Acc : {valid_acc*100:.2f}%')

