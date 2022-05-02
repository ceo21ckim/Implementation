"""
1. importing modules
2. loading data 
3. tokenizing data 
4. creating data splits
5. creating a vocabulary
6. numericalizing data 
7. creating the dataloaders 
"""

# 1. Importing modules 
import functools, sys
import datasets
import torch
import torchtext

import matplotlib.pyplot as plt 
import numpy as np 
import torch.nn as nn 
import torch.optim as optim 
import tqdm 



import spacy
# spacy.cli.download('en_core_web_sm')


seed = 0 
torch.manual_seed(seed)
np.random.seed(seed)

train_data, test_data = datasets.load_dataset('imdb', split=['train', 'test'])

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

test_sent = "Hello world! How are you doing today? I'm doing fantastic!"
tokenizer(test_sent)

def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    return {'tokens':tokens}

max_length = 256 

train_data = train_data.map(tokenize_example, fn_kwargs={'tokenizer':tokenizer, 'max_length':max_length})
test_data = test_data.map(tokenize_example, fn_kwargs={'tokenizer':tokenizer, 'max_length':max_length})

test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data['train']
valid_data = train_valid_data['test']

min_freq = 5
special_tokens = ['<unk>', '<pad>']
vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'], 
                                                  min_freq=min_freq, 
                                                  specials=special_tokens)

# index to string
vocab.get_itos()[:10]

unk_index = vocab['<unk>']
pad_index = vocab['<pad>']

unk_index, pad_index

vocab.set_default_index(unk_index)

def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids':ids}

train_data = train_data.map(numericalize_data, fn_kwargs={'vocab':vocab})
valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab':vocab})
test_data = test_data.map(numericalize_data, fn_kwargs={'vocab':vocab})

# if you input that not in vocab, output 0
vocab['fdnsf']

# to tensor
train_data = train_data.with_format(type='torch', columns = ['ids', 'label'])
valid_data = valid_data.with_format(type='torch', columns = ['ids', 'label'])
test_data = test_data.with_format(type='torch', columns = ['ids', 'label'])

def collate(batch, pad_index):
    batych_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'ids':batch_ids, 
             'label': batch_label}
    return batch

batch_size = 512

from torch.utils.data import DataLoader

collate = functools.partial(collate, pad_index=pad_index)
train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate)
test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate)

class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super(NBoW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, ids):
        # ids = [batch size, sequence len]
        embedding = self.embedding(ids)
        #embedding = [batch_size, sequence len, embedding dim]
        pooled = embedding.mean(dim=1)
        #pooled = [batch_size, embedding dim]
        prediction = self.fc(pooled)
        #prediction = [batch_size, output dim]
        return prediction

vocab_size = len(vocab)
embedding_dim = 300
output_dim = len(train_data.unique('label'))

model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The Model has {count_parameters(model):,} trainable parameters')

vectors= torchtext.vocab.FastText()

hello_vector = vectors.get_vecs_by_tokens('hello')
hello_vector.shape 

pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
pretrained_embedding.shape 

model.embedding.weight 

pretrained_embedding 

model.embedding.weight.data = pretrained_embedding 

model.embedding.weight 

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device 

model = model.to(device)
criterion = criterion.to(device)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape 
    predicted_classes = prediction.argmax(dim=1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size 
    return accuracy

def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []

    for batch in tqdm.tqdm(dataloader, desc='traing...', files=sys.stdout):
        ids = batch['ids'].to(device)
        label = batch['label'].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss)
        epoch_accs.append(accuracy)

    return epoch_losses, epoch_accs

def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluting...', file=sys.stdout):
            ids = batch['ids'].to(device)
            label = batch['label'].to(device)
            prediction = model(ids)
            loss = criterion(ids, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss)
            epoch_accs.append(accuracy)
    return epoch_losses, epoch_accs 

n_epochs = 10
best_valid_loss = float('inf')

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

for epoch in range(n_epochs):
    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
    valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)

    train_losses.extend(train_loss)
    train_accs.extend(train_acc)
    valid_losses.extend(valid_loss)
    valid_accs.extend(valid_acc)

    epoch_train_loss = np.mean(train_losses)
    epoch_train_acc = np.mean(train_accs)
    epoch_valid_loss = np.mean(valid_losses)
    epoch_valid_acc = np.mean(valid_accs)

    if epoch_valid_loss < best_valid_loss:
        best_valid_loss = epoch_valid_loss 
        torch.save(model.state_dict(), 'nbow.pt')

    print(f'epoch: {epoch+1}')
    print(f'train_loss : {epoch_train_loss:.3f}, train_acc : {epoch_train_acc:.3f}')
    print(f'valid_loss : {epoch_valid_loss:.3f}, valid_acc : {epoch_valid_acc:.3f}')