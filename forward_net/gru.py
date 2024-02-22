from torch import nn
from torchvision.ops import MLP
import torch
class GRUModel(nn.Module):
    def __init__(self, emb_dim=512, hidden_dim=32, n_layers=1, dropout=0.1, other_inp_shape=131,
    mlp_hid_dim: int = 32, mlp_n_layers:int = 1, output_size: int = 131 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, n_layers, bidirectional=True, dropout=0.0 if n_layers==1 else dropout,
        batch_first=True)
        # self.mlp = MLP(2*hidden_dim + other_inp_shape, [mlp_hid_dim]*mlp_n_layers+[output_size],
        # activation_layer=nn.ReLU, dropout=dropout, inplace=False)
        self.mlp = MLP(2*hidden_dim + other_inp_shape, [mlp_hid_dim]*mlp_n_layers,
        activation_layer=nn.LeakyReLU, dropout=dropout, inplace=False)
        # Try to remove output size from mlp and add this fc
        # in this way we won't have dropout as the last layer
        self.fc = nn.Linear(mlp_hid_dim, output_size)
        self.init_weights()
        #self.dropout = nn.Dropout(dropout)
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.normal_(param.data, mean=0, std=0.01)
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            else:
                nn.init.constant_(param.data, 0)

    def forward(self, x, src_len, other_input):
        # x = self.embedding(x)
        # x = self.dropout(x)
        
        # pack the sequence before passing to RNN
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_len, batch_first=True,
                                                    enforce_sorted=True)
        
        # packed_outputs, hidden_state = self.gru(packed_x)
        _, hidden_state = self.gru(packed_x)
        
        # pad packed output sequence
        #outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        # Combine the forward and backward RNN's hidden states to be input to decoder
        hidden_state = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        output = torch.cat([hidden_state, other_input], dim=1) 
        #hidden_state = torch.tanh(self.fc())
        output = self.mlp(output)
        output = nn.functional.leaky_relu(output, inplace=False)
        output = self.fc(output)
        return output
        #return outputs, hidden_state
class GRUModelNoMLP(nn.Module):
    '''This model is used to test the performance of the model without MLP'''
    def __init__(self, emb_dim=512, hidden_dim=32, n_layers=1, dropout=0.1, other_inp_shape=131,
    mlp_hid_dim: int = 32, mlp_n_layers:int = 1, output_size: int = 131) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(emb_dim, hidden_dim, n_layers, bidirectional=True, dropout=0.0 if n_layers==1 else dropout,
        batch_first=True)
        self.fc = nn.Linear(2*hidden_dim, output_size)
        #self.init_weights()
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.normal_(param.data, mean=0, std=0.01)
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            else:
                nn.init.constant_(param.data, 0)
    def forward(self, x, src_len, other_input):
        # x = self.embedding(x)
        # x = self.dropout(x)
        
        # pack the sequence before passing to RNN
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_len, batch_first=True,
                                                    enforce_sorted=True)
        
        # packed_outputs, hidden_state = self.gru(packed_x)
        _, hidden_state = self.gru(packed_x)
        
        # pad packed output sequence
        #outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        # Combine the forward and backward RNN's hidden states to be input to decoder
        hidden_state = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        output = self.fc(hidden_state)
        return output

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# Test model:
if __name__ == '__main__':
    batch, seq_len, embed, other_inp_shape = 2, 3, 5, 7
    model = GRUModel(emb_dim=embed, other_inp_shape=other_inp_shape)
    #model = GRUModelNoMLP(emb_dim=embed, other_inp_shape=other_inp_shape)
    print(model)
    print(count_parameters(model))
    src = torch.rand((batch, seq_len, embed))
    other_inp = torch.rand((batch, other_inp_shape))
    src_len = [seq_len]*batch
    out = model(src, src_len , other_inp)
    print(out.shape)

'''
# class Attention(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.attention = nn.Linear(3*hidden_dim, hidden_dim)
#         self.v = nn.Linear(hidden_dim, 1, bias=False)
    
#     def forward(self, hidden_state, encoder_outputs, mask):
#         batch_size = encoder_outputs.shape[1]
#         src_len = encoder_outputs.shape[0]
        
#         # repeat decoder hidden state src_len times
#         hidden_state = hidden_state.unsqueeze(1).repeat(1, src_len, 1)
#         encoder_outputs = encoder_outputs.permute(1, 0, 2)

#         energy = torch.tanh(self.attention(torch.cat((hidden_state, encoder_outputs), dim = 2))) 
#         attention = self.v(energy).squeeze(2)
#         # apply the mask so that the model don't pay the attention to paddings.
#         # it's applied before softmax so that the masked values which are very
#         # small will be zero'ed out after softmax
#         attention = attention.masked_fill(mask == 0, -1e10)
        
#         return F.softmax(attention, dim=1)

# class Decoder(nn.Module):
#     def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, attention, dropout=0.2):
#         super().__init__()
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.attention = attention
#         self.n_layers = n_layers
#         self.embedding = nn.Embedding(output_dim, emb_dim)
#         self.gru = nn.GRU(emb_dim+(hidden_dim*2), hidden_dim, n_layers, dropout=0.0 if n_layers==1 else dropout)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_dim*3 + emb_dim, output_dim)
    
#     def forward(self, input, hidden, encoder_outputs, mask):
#         input = input.unsqueeze(0)
        
#         #input = [1, batch size]
        
#         embedded = self.dropout(self.embedding(input))
        
#         #embedded = [1, batch size, emb dim]
        
#         a = self.attention(hidden, encoder_outputs, mask)
                
#         #a = [batch size, src len]
        
#         a = a.unsqueeze(1)
        
#         #a = [batch size, 1, src len]
        
#         encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
#         #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
#         weighted = torch.bmm(a, encoder_outputs)
        
#         #weighted = [batch size, 1, enc hid dim * 2]
        
#         weighted = weighted.permute(1, 0, 2)
        
#         #weighted = [1, batch size, enc hid dim * 2]
        
#         rnn_input = torch.cat((embedded, weighted), dim = 2)
        
#         #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
#         output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))
        
#         #output = [seq len, batch size, dec hid dim * n directions]
#         #hidden = [n layers * n directions, batch size, dec hid dim]
        
#         #seq len, n layers and n directions will always be 1 in this decoder, therefore:
#         #output = [1, batch size, dec hid dim]
#         #hidden = [1, batch size, dec hid dim]
#         #this also means that output == hidden
#         assert (output == hidden).all()
        
#         embedded = embedded.squeeze(0)
#         output = output.squeeze(0)
#         weighted = weighted.squeeze(0)
        
#         prediction = self.fc(torch.cat((output, weighted, embedded), dim = 1))
        
#         #prediction = [batch size, output dim]
        
#         # also return attention vector as it's required at inference time to vizualize
#         # attention mechanism.
#         return prediction, hidden.squeeze(0), a.squeeze(1)

# class EncoderDecoder(nn.Module):
    # def __init__(self, encoder, decoder, pad_idx):
    #     super().__init__()
    #     self.encoder = encoder
    #     self.decoder = decoder
    #     self.pad_idx = pad_idx
        
    #     assert self.encoder.hidden_dim == decoder.hidden_dim
    #     assert self.encoder.n_layers == decoder.n_layers
    
    # def _create_mask(self, x):
    #     return (x != self.pad_idx).permute(1, 0)
    
    # def forward(self, x, src_len, y, teacher_forcing_ratio=0.75):
        
    #     target_len = y.shape[0]
    #     batch_size = y.shape[1]
    #     target_vocab_size = self.decoder.output_dim  # Output dim
        
    #     outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        
    #     # Encode the source text using encoder. Last hidden state of encoder is context vector.
    #     encoder_outputs, hidden_state = self.encoder(x, src_len)
        
    #     # First input is <sos>
    #     input = y[0,:]
        
    #     mask = self._create_mask(x)
        
    #     # Decode the encoded vector using decoder
    #     for t in range(1, target_len):
    #         # attention vector returned by decoder is not required while training
    #         output, hidden_state, _ = self.decoder(input, hidden_state, encoder_outputs, mask)
    #         outputs[t] = output
    #         teacher_force = random.random() < teacher_forcing_ratio
    #         pred = output.argmax(1)
    #         input = y[t] if teacher_force else pred
        
    #     return outputs
'''