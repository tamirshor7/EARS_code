import torch
import torch.nn as nn
from EARS.localization.transformer_encoder import PatchEmbed

class Encoder(nn.Module):
    '''
    Embed audio using a transformer encoder.
    '''
    def __init__(self, input_channels:int=8, hidden_dim:int = 1024, num_head:int=1, dropout:float=0.0, num_layers:int=1) -> None:
        super().__init__()
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=d_model,
        #         nhead=num_head,
        #         dim_feedforward=dim_feedforward,
        #         dropout=dropout,
        #         batch_first=True
        #     ),
        #     num_layers=num_layers
        # )
        self.proj = PatchEmbed(num_channels=input_channels, embed_dim=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_dim, num_head, hidden_dim, dropout, batch_first=True),
                num_layers=num_layers,
            )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    '''
    Decode audio using a transformer decoder.
    '''
    def __init__(self, d_model:int = 8, num_head:int=1, dim_feedforward:int=2048, dropout:float=0.0, num_layers:int=1) -> None:
        super().__init__()
        # self.transformer = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(
        #         d_model=d_model,
        #         nhead=num_head,
        #         dim_feedforward=dim_feedforward,
        #         dropout=dropout,
        #         batch_first=True
        #     ),
        #     num_layers=num_layers
        # )
        self.decoder = nn.TransformerDecoder
    def forward(self, x):
        return self.transformer(x)

class AutoEncoder(nn.Module):
    '''
    Project audio into a lower dimensional space and then reconstruct it.
    '''
    def __init__(self, d_model:int=8, num_head:int=1, dim_feedforward:int=2048, dropout:float=0.0, num_layers:int=1) -> None:
        super().__init__()
        self.encoder = Encoder(d_model, num_head, dim_feedforward, dropout, num_layers)
        self.decoder = Decoder(d_model, num_head, dim_feedforward, dropout, num_layers)
    def forward(self, x):
        initial_shape = x.shape
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(initial_shape)
        return x
    
if __name__ == '__main__':
    batch_size, t, embedding_dim, hidden_dim, num_layers, num_head = 32, 10, 8, 16, 1, 1
    x = torch.randn(batch_size, t, embedding_dim)
    encoder = Encoder(embedding_dim, num_head, hidden_dim, num_layers)
    decoder = Decoder(embedding_dim, num_head, hidden_dim, num_layers)
    print(encoder)
    print(decoder)
    print(f"input shape: {x.shape}")
    x = encoder(x)
    print(f"after encoder: {x.shape}")
    x = decoder(x)
    print(f"after decoder: {x.shape}")
