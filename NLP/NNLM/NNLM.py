# n-gram 기반 언어 모델의 약점을 보완하는 신경망 언어 모델 NNLM
# n-gram 기반 언어 모델은 간편하지만 훈련 데이터에서 보지 못한 단어의 조합에는 꽤 취약한 부분이 있음.
# e.g. 고양이는 좋은 반려동물입니다.
# P(반려동물|강아지는, 좋은) 
# P(반려동물|자동차는, 좋은)

# 사람은 위 두 문장을 파악할 수 있지만, n-gram based model은 '강아지는' 이라는 단어가 출현하지 않기 때문에 확률을 구할 수 없음.
 
# NNLM : neural network Language model
# RNNLM : RNN-based NNLM model
# !pip install data_loader


import torch 
import torch.nn as nn 
import data_loader

class LanguageModel(nn.Module):
    def __init__(self,
                 vocab_size, 
                 word_vec_dim = 512, 
                 hidden_size = 512, 
                 n_layers = 4, 
                 drop_out_p = .2, 
                 max_length = 255
                 ):
        
        self.vocab_size = vocab_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size 
        self.n_layers = n_layers
        self.drop_out_p = drop_out_p
        self.max_length = max_length
        
        super(LanguageModel, self).__init__()
        
        self.emb = nn.Embedding(vocab_size, 
                                word_vec_dim, 
                                padding_idx = data_loader.PAD
                                )
        
        self.rnn = nn.LSTM(word_vec_dim, 
                          hidden_size, 
                          n_layers,
                          batch_first = True, 
                          dropout = drop_out_p
                          )
        
        self.out = nn.Linear(hidden_size, vocab_size, bias = True)
        self.log_softmax = nn.LogSoftmax(dim = 2)
        
    def forward(self, x):
        x = self.emb(x)
        
        x, (h, c) = self.rnn(x)
        
        x = self.out(x)
        
        y_hat = self.log_softmax(x)
        
    def search(self, batch_size = 64, max_length = 255):
        x = torch.LognTensor(batch_size, 1).to(next(self.parameters()).device).zero_() + data_loader.BOS 
        
        is_undone = x.new_ones(batch_size, 1).float()
        
        y_hats, indice = [], []
        n, c = None, None
        while is_undone.sum() > 0 and len(indice) < max_length :
            x = self.emb(x)
            
            x, (h, c) = self.rnn(x, (h, c)) if h is not None and c is not None else self.rnn(x)
            
            y_hat = self.log_softmax(x)
            
            y_hat += [y_hat]
            
            y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)
            
            y = y.masked_fill_((1. -is_undone).byte(), data_loader.PAD)
            is_undone = is_undone * torch.ne(y, data_loader.EOS).float()
            
            indice += [y]
            
            x = y
            
        y_hat = torch.cat(y_hats, dim = 1)
        indice = torch.cat(indice, dim = -1)
        
        
        return y_hats, indice