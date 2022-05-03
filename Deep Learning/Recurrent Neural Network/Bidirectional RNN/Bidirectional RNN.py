# Bidirectional RNN
# As well as having an RNN processing the words in the sentence from the first to the last (a forward RNN)
# We have a second RNN processing the words in the sentence from the last to the first (a backword RNN)

# torch==1.11.0 
# torchtext==0.11.0

import random

import torch
import torch.nn as nn 
from torchtext.legacy import datasets
from torchtext.legacy import data

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cuda.deterministic = True



LABEL = data.LabelField(dtype = torch.float)
TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  include_lengths = True)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))

MAX_VOCAB_SIZE = 25_000 

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = 'glove.6B.100d', unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, sort_within_batch=True, device=device)



class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers = n_layers, 
            bidirectional = bidirectional, 
            dropout = dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text, text_length):
        # text = [sent length, batch size]
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent length, batch size, emb dim]

        # pack sequence 
        # length need to be 'cpu'
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence 
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hidden dim * num directions] if bidirections -> hidden dim * 2
        # output over padding tokens are zero tensors 

        # hidden = [num layers * num directions, batch size, hidden dim]
        # cell = [num layers * num directions, batch size, hidden dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers 
        # and apply dropout 

        hidden = self.dropout(torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim = 1))

        return self.fc(hidden)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100 
HIDDEN_DIM = 256 
OUTPUT_DIM = 1
N_LAYERS = 2 
BIDIRECTION = True 
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(
    INPUT_DIM, 
    EMBEDDING_DIM, 
    HIDDEN_DIM, 
    OUTPUT_DIM, 
    N_LAYERS, 
    BIDIRECTION, 
    DROPOUT, 
    PAD_IDX 
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors 
print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)

# like initializing the embeddings, this should be done on the weight.data and not the weight
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

###################################### Train the Model ########################################

import torch.optim as optim 

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc 

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0 

    model.train()
    for batch in iterator:
        
        optimizer.zero_grad()
        text, text_length = batch.text 
        predictions = model(text, text_length).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluation(model, iterator, criterion):
    epoch_loss, epoch_acc = 0, 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_length = batch.text 
            predictions = model(text, text_length).squeeze(1)

            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time 
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time 
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 5 

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluation(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss :
        best_valid_loss = valid_loss 
        torch.save(model.state_dict(), 'Bidirectional RNN.pt')

    print(f'Epoch : {epoch+1:02} | Epoch Time : {epoch_mins}m, {epoch_secs}s')
    print(f'\t Train Loss : {train_loss:.3f} | Train Acc : {train_acc*100:.2f}%')
    print(f'\t Val. Loss : {valid_loss:.3f}  | Val. Acc : {valid_acc*100:.2f}%')

