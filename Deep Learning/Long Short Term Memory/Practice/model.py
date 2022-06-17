import typing 
import torch 
import torch.nn as nn 

class lstm(nn.Module):
    def __init__(self, vocab_size : int, embedding_dim : int, hidden_dim : int, output_dim : int, n_layers : int, bidirectional : bool, dropout_rate : float) -> None:
        super(lstm, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedding = self.embedding(ids)

        if self.dropout : 
            embedding = self.dropout(embedding)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedding, length, batch_first=True, enforce_sorted = False)

        o, (h, c) = self.lstm(packed_embedded)

        output, output_length = nn.utils.rnn.pad_packed_sequence(o)

        pred = self.fc(h)
        return pred 



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        
if __name__ == '__main__':
    VOCAB_SIZE = 8000
    EMBED_DIM = 768
    HIDDEN_DIM = 128
    OUTPUT_DIM = 2
    LAYERS = 2 
    BIDIREC = True 
    DROPOUT = 0.5
    model = lstm(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, LAYERS, BIDIREC, DROPOUT)
    input_ids = torch.LongTensor([[100, 70, 88, 250]])
    model(input_ids, VOCAB_SIZE)

embed = model.embedding(input_ids)
output, (h, c) = model.lstm(embed)

output.shape
h.shape
c.shape


torch.cat([h[-1], h[-2]], dim = -1).shape
h[-1].shape