# pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# torchtext==0.11.0
# pip install spacy 

"""
We'll be using a Recurrent Neural Network (RNN) as they are commonly used in analysing sequences. An RNN tasks in sequence of words. (NNLM)
h_t = RNN(x_t, h_t-1)
Once we have our final hidden state (h_T), we feed it through a linear layer f(Fully Connected Layer (FC Layer))
h_0: initial hidden state is a tensor initialized to all zeros.
"""
import torch
from torchtext.legacy import data 
import sys, random
sys.path.append('../')

SEED = 1234 
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

"""
'Field' define how your data should be processed. In our sentiment classification task the data consists of both the raw string of the review and the sentiment, either 'pos' or 'neg'.
TEXT field has tokenize='spacy' as an argument. Consequently, can use spaCy tokenizer.
LABEL defined by a LabelField, a special subset of the Field class specifically used for handling labels. We will explain the dtype argument later.
"""
###################################  Preparing Data  #########################################
import spacy
spacy.cli.download('en_core_web_sm')
TEXT = data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float)

# used IMDB datasets
from torchtext.legacy import datasets 
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL) # train_test_split method 

print(f'Number of training examples: {len(train_data)}') ; print(f'Number of testing examples: {len(test_data)}')


# We can also check an example.
# vars(): return __dict__ attribute.
print(vars(train_data.examples[0]))


"""
split_ratio : default 7:3, we can change the ratio of the split, i.e. a split_ratio of 0.8 would mean 80% of the examples make up the training set and 20% make up the validation set.
We also pass our random seed to the random_state argument, ensuring that we get the same train/validation split each time.
"""
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f'Number of training examples: {len(train_data)}') ; print(f'Number of validation examples: {len(valid_data)}') ; print(f'Number of testing examples: {len(test_data)}')

"""
we have to build a vocabulary. This is a effectively a look up table where every unique word in your data set has a corresponding index (an integer).
we do this asour machine learning model cannot operate on strings, only numbers, Each index is used to construct a one-hot vector for each word. 
A one-hot vector where all of the elements are 0, except one whice is 1, and dimensionality is the total number of unique words in your vocabulary. 
if vocabulary size is 100,000, which means that our one-hot vectors will have 100,000 dimension.
"""

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

"""
TEXT.vocab size is 25002, because of One of the addition tokens is the <UNK> token and the other is a <PAD> token.
LABEL.vocab included 'pos' or 'neg'
"""
print(f'Unique token in TEXT vocabulary: {len(TEXT.vocab)}') ; print(f'Unique token in LABEL vocabulary: {len(LABEL.vocab)}')
print(TEXT.vocab.freqs.most_common(n=20))

# can also see the vocabulary directly using either the 'stoi' (string to int) or 'itos' (int to string) method.
print(TEXT.vocab.itos[:10])
print(TEXT.vocab.stoi['hi'])

BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
We'll use a 'BucketIterator' which is a special type of iterator that will return a batch of examples where each example is of s aimilar length, 
minimizing the amount of padding per example.
"""

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size = BATCH_SIZE, device = device)

###################################  Build the Model  #########################################
"""
Within the '__init__' we define the layers of the module. Oure thre layers are an embedding layer, our RNN, and a linear layer. 
All layers have their parameters initialized to random values, unless explicitly specified.
The embedding layer is used to transform our sparse one-hot vector (sparce as most of the elements are 0) into a dense embedding vector.
The embedding layer is simply a single fully conneted layer. As weel as reducing the dimensionality of the input to the RNN, there is the theory that words which
have similar impact on the sentiment of the review are mapped close together in this dense vector space. 
Finally, the linear layer takes the final hidden state and feed it through a fully conneted layer, tranforming it to the correct output diemension.
Each batch, 'text', is a tensor of size [sentence length, batch size]. That is a batch of sentences, each having each word converted into a one-hot vector.
You may notice that this tensor should have another dimension due to the one-hot vectors, however PyTorch conveniently stores a noe-hot vector as it's index values,
i.e. the tensor representing a sentence is just a tensor of the indexes for each token in that sentence. 
The act of convering a list of tokens into a list of indexes is commonly called numericalizing.
The input batch is then passed through the embedding layer to get 'embedded', which gives us a dense vector representation of our sentences, 
'embedded' is a tensor of size [sentence length, batch size, embedding dim].
The RNN returns tensors, 'output' of size [sentence length, batch size, hidden dim] and 'hidden' of size [1, batch size, hidden dim]. 
'output' is the concatenation of the hidden state from every time step, whereas 'hidden' is simply the final hidden state. We verify this using the 'assert' statetment.
"""

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        
        # output : [sentence length, batch size, hidden dim]
        # hidden : [1, batch size, hidden dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0)) # squeeze: which is used to remove a dimension of size 1.


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100 
HIDDEN_DIM = 256 
OUTPUT_DIM = 1
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)


# We'll tell us how many trainable parameters our model hase so we can compare the number of parameters across different models.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

###################################  Train the Model  #########################################
import torch.optim as optim 
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuarcy(pred_y, true_y):
    """
    if used self.Sigmoid() in __init__ , Then use pred_y instead of torch.sigmoid(pred_y)
    """
    rounded_preds = torch.round(torch.sigmoid(pred_y))
    correct = (rounded_preds == true_y).float()
    acc = correct.sum()/len(correct)
    return acc 

"""
'model.train()' is used to put the model in 'training mode', which turns on dropout and batch normalization.
For each batch, we first zero the gradients. Each parameter in a model has a grad attribute which stores the gradient calcuated by the criterion.
PyTorch dose not automatically remove (or 'zero') the gradients calculated from the last gradient calculation, so they must be manually zeroed.
We then feed the batch of sentences into the model. Note, you do not need to do model.forward(batch.text).
The squeeze is needed as the predictions are initally size [batch size, 1], and we need to remove the dimension of size 1 as PyTorch expects the predictions
input to our criterion function to be size [batch size].
The loss and accuracy is accumulated across the epoch, the 'item' method is used to extract a scalar from a tensor which only contains a single value.
You may recall when initializing the 'LABEL' field, we set 'dtype=torch.float'. This is because TorchText sets tensors to be 'LongTensor' by default,
however our criterion expects both inputs to be 'FloatTensor' 
"""

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0 

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        pred = model(batch.text).squeeze(1) # [batch size, 1] -> [batch size]
        loss = criterion(pred, batch.label)

        acc = binary_accuarcy(pred, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)

"""
'evaluate' is similar to 'train', with a few modifications as you don't want to update the parameters when evaluating.
'model.eval()' puts the model in 'evaluation mode', this turns off dropout and batch normalization. 
No gradients are calculated on PyTorch operations inside the 'with no_grad()' block, This causes less memory to be used and speeds up computation.
The rest of the function is the same as 'train', with the removal of 'optimizer.zero_grad()', loss.backward()' and 'optimizer.step()', 
as we do not update the model's parameters when evaluating.
"""

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0 

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            pred = model(batch.text).squeeze(1) # [batch size, 1] -> [batch size]
            loss = criterion(pred, batch.label)

            acc = binary_accuarcy(pred, batch.label)

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
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss 
        torch.save(model.state_dict(), 'NNLM with RNN.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')



# evaluate
model.load_state_dict(torch.load('NNLM with RNN.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')