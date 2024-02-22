import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
import torch.nn.functional as F
import numpy as np

#from transformers import ASTModel
 
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 4_000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        #pe = torch.zeros(max_len, 1, d_model)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        Returns:
            output Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        seq_cut = min(x.size(1), self.pe.size(1))
        x = x + self.pe[:,:seq_cut,:]
        # Given that we are giving as input the raw audio data
        # and we are not embedding it, we should concatenate
        # the positional encoding to the input rather than simply summing it!
        #x = torch.cat((x, self.pe[:x.size(0)]))
        return x

class PatchEmbed(nn.Module):
    '''Class taken from ASTModel'''
    #def __init__(self, img_size=224, patch_size=16, in_chans=8, embed_dim=768):

    # before max_length was 300
    def __init__(self, num_channels:int=8, embed_dim:int=1024, max_length:int=140_000) -> None:
        super().__init__()

        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.num_patches = num_patches

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        kernel_size = (3,3,2)
        self.proj = nn.Conv3d(in_channels=num_channels, out_channels=embed_dim, 
                              kernel_size=kernel_size, stride=kernel_size)
        # add position embedding to the patch embedding
        # inject information about the microphone
        self.channel_positional_embedding = nn.Parameter(torch.randn(1, embed_dim, 1), requires_grad=True)
        # inject information about the time
        self.max_length = max_length
        self.time_positional_embedding = nn.Parameter(torch.randn(1,1, max_length), requires_grad=True)

    def forward(self, x):
        x = self.proj(x).squeeze()
        x = x.flatten(-2,-1)
        cut = min(x.shape[-1], self.max_length)
        x = x + self.time_positional_embedding[:,:, :cut]
        x = x + self.channel_positional_embedding
        #x = x.flatten(1).unsqueeze(-1)
        x = x.transpose(1,2)
        return x
    
class SpectrogramTransformer(nn.Module):
    def __init__(self, hidden_dim:int=1024, num_layers:int=1, num_heads:int=1, dropout:float=0.1, num_coordinates:int = 1):
        super().__init__()
        self.sound_projection = PatchEmbed(num_channels=8, embed_dim=hidden_dim)
        self.sound_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )

        self.modulation_projection = PatchEmbed(num_channels=4, embed_dim=hidden_dim)
        
        self.modulation_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )
        self.linear = nn.Linear(2*hidden_dim, num_coordinates)

    def forward(self, x, phi):
        x = self.sound_projection(x)
        x = self.sound_transformer(x)
        x = x.mean(dim=1)

        phi = self.modulation_projection(phi)
        phi = self.modulation_transformer(phi)
        phi = phi.mean(dim=1)

        phi = phi.expand(x.shape[0], -1)
        x = torch.cat((x, phi), dim=-1)

        x = self.linear(x)
        return x

class AggregateTransformer(nn.Module):
    def __init__(self, hidden_dim:int=1024, num_layers:int=1, num_heads:int=1, dropout:float=0.1, num_coordinates:int = 1):
        super().__init__()
        self.sound_projection = PatchEmbed(num_channels=8, embed_dim=hidden_dim)
        self.sound_transformer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True)

        self.modulation_projection = PatchEmbed(num_channels=4, embed_dim=hidden_dim)
        self.modulation_transformer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True)

        self.aggregate_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )
        self.linear = nn.Linear(hidden_dim, num_coordinates)
    
    def forward(self, sound, phi):
        sound = self.sound_projection(sound)
        sound = self.sound_transformer(sound)        
        sound = sound.mean(dim=1)

        phi = self.modulation_projection(phi)
        phi = self.modulation_transformer(phi)
        phi = phi.mean(dim=1)
        phi = phi.expand(sound.shape[0], -1)

        aggregated = sound + phi
        aggregated = self.aggregate_transformer(aggregated)
        output = self.linear(aggregated)
        return output

class SoundTransformer(nn.Module):
    def __init__(self, hidden_dim:int=1024, num_layers:int=1, num_heads:int=1, dropout:float=0.1, num_coordinates:int = 1):
        super().__init__()
        self.sound_projection = PatchEmbed(num_channels=8, embed_dim=hidden_dim)
        self.sound_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )
        self.linear = nn.Linear(hidden_dim, num_coordinates)
    def forward(self, sound, phi):
        sound = self.sound_projection(sound)
        sound = self.sound_transformer(sound)
        sound = sound.mean(dim=1)
        output = self.linear(sound)
        return output  


class OrientationAggregateTransformer(nn.Module):
    def __init__(self, hidden_dim:int=1024, num_layers:int=1, num_heads:int=1, dropout:float=0.1, num_coordinates:int = 1):
        super().__init__()
        self.sound_projection = PatchEmbed(num_channels=8, embed_dim=hidden_dim)
        self.sound_transformer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True)

        self.modulation_projection = PatchEmbed(num_channels=4, embed_dim=hidden_dim)
        self.modulation_transformer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True)

        self.orientation_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        self.aggregate_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )
        self.linear = nn.Linear(hidden_dim, num_coordinates)
    
    def forward(self, sound, phi, orientation):
        sound = self.sound_projection(sound)
        sound = self.sound_transformer(sound)        
        sound = sound.mean(dim=1)

        phi = self.modulation_projection(phi)
        phi = self.modulation_transformer(phi)
        phi = phi.mean(dim=1)
        phi = phi.expand(sound.shape[0], -1)

        orientation = self.orientation_mlp(orientation)

        aggregated = sound + phi + orientation
        aggregated = self.aggregate_transformer(aggregated)
        output = self.linear(aggregated)
        return output
    
class OrientationAggregateTransformerMLP(nn.Module):
    def __init__(self, hidden_dim=1024, num_layers=1, num_heads=1, dropout=0.1, num_coordinates=1):
        super().__init__()
        self.sound_projection = PatchEmbed(num_channels=8, embed_dim=hidden_dim)
        self.sound_transformer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True)

        self.modulation_projection = PatchEmbed(num_channels=4, embed_dim=hidden_dim)
        self.modulation_transformer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True)

        self.orientation_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.aggregate_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )
        self.mlp_aggregated = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # ATTENTION: leave the linear layer in the last place, otherwise you need to select the parameters of the linear layer (if you move the linear layer, then the index of their parameters will be different and it needs to be found again) in
        #           EARS/localization/train_pipeline.py/optimizer_load_state_dict() and in EARS/localization/train_separate.py/optimizer_load_state_dict()
        #           if len(state_dict['param_groups']) == 2 and if args.aggregator == 'deep'
        self.linear = nn.Linear(hidden_dim, num_coordinates)

    def forward(self, sound, phi, orientation):
        sound = self.sound_projection(sound)
        sound = self.sound_transformer(sound)
        sound = sound.mean(dim=1)

        phi = self.modulation_projection(phi)
        phi = self.modulation_transformer(phi)
        phi = phi.mean(dim=1)
        phi = phi.expand(sound.shape[0], -1)

        orientation = self.orientation_mlp(orientation)

        # Aggregation with concatenation
        aggregated = torch.cat((sound, phi, orientation), dim=1)
        aggregated = self.mlp_aggregated(aggregated)
        aggregated = self.aggregate_transformer(aggregated)
        output = self.linear(aggregated)
        return output
    
class OrientationAggregateTransformerAttention(nn.Module):
    def __init__(self, hidden_dim=1024, num_layers=1, num_heads=1, dropout=0.1, num_coordinates=1):
        super().__init__()
        self.sound_projection = PatchEmbed(num_channels=8, embed_dim=hidden_dim)
        self.sound_transformer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True)

        self.modulation_projection = PatchEmbed(num_channels=4, embed_dim=hidden_dim)
        self.modulation_transformer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True)

        self.orientation_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.aggregate_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(3*hidden_dim, num_heads, 3*hidden_dim, dropout, batch_first=True),
            num_layers
        )

        self.linear = nn.Linear(3*hidden_dim, num_coordinates)

    def forward(self, sound, phi, orientation):
        sound = self.sound_projection(sound)
        sound = self.sound_transformer(sound)
        sound = sound.mean(dim=1)

        phi = self.modulation_projection(phi)
        phi = self.modulation_transformer(phi)
        phi = phi.mean(dim=1)
        phi = phi.expand(sound.shape[0], -1)

        orientation = self.orientation_mlp(orientation)

        # Aggregation with concatenation
        aggregated = torch.cat((sound, phi, orientation), dim=1)
        aggregated = self.aggregate_transformer(aggregated)
        output = self.linear(aggregated)
        return output

    

class OrientationSpectrogramTransformer(nn.Module):
    def __init__(self, hidden_dim:int=1024, num_layers:int=1, num_heads:int=1, dropout:float=0.1, num_coordinates:int = 1):
        super().__init__()
        self.sound_projection = PatchEmbed(num_channels=8, embed_dim=hidden_dim)
        self.sound_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )

        self.modulation_projection = PatchEmbed(num_channels=4, embed_dim=hidden_dim)
        
        self.modulation_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )

        self.orientation_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        self.linear = nn.Linear(3*hidden_dim, num_coordinates)

    def forward(self, x, phi, orientation):
        x = self.sound_projection(x)
        x = self.sound_transformer(x)
        x = x.mean(dim=1)

        phi = self.modulation_projection(phi)
        phi = self.modulation_transformer(phi)
        phi = phi.mean(dim=1)

        orientation = self.orientation_mlp(orientation)

        phi = phi.expand(x.shape[0], -1)
        x = torch.cat((x, phi, orientation), dim=-1)

        x = self.linear(x)
        return x

def softmax(input, t=1.0):
    ex = torch.exp(input/t)
    sum = torch.sum(ex, axis=1)
    return ex / sum



class Fire(nn.Module):
    "A block that follows the Fire block in SqueezeNet."
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class CNNEncoder(nn.Module):
    "a CNN for audio localization that follows the SqueezeNet architecture."
    def __init__(self, num_channels=8, hidden_dim=64) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=64,
                               kernel_size=(3,3,2), stride=(3,3,2))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire2 = Fire(64, 16, 64, 64)
        #self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fire5 = Fire(256, 32, 128, 128)
        # self.fire6 = Fire(256, 48, 192, 192)
        # self.fire7 = Fire(384, 48, 192, 192)

        # self.fire8 = Fire(384, 64, 256, 256)
        self.fire8 = Fire(256, 64, 256, 256)

        self.maxpool8 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fire9 = Fire(512, 64, 256, 256)
        self.dropout9 = nn.Dropout(p=0.5)
        # self.conv10 = nn.Conv2d(512, 12, kernel_size=1)
        # self.relu10 = nn.ReLU(inplace=True)
        # self.avgpool10 = nn.AdaptiveAvgPool2d((1,num_coordinates))
        # self.softmax10 = nn.Softmax(dim=-1)
        self.conv10 = nn.Conv2d(512, 12, kernel_size=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.avgpool10 = nn.AdaptiveAvgPool3d((1,1,hidden_dim))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.squeeze(-1)
        x = self.maxpool1(x)
        x = self.fire2(x)
        # x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)
        x = self.fire5(x)
        # x = self.fire6(x)
        # x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)
        x = self.fire9(x)
        x = self.dropout9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.avgpool10(x)
        x = x.squeeze(1).squeeze(1)
        return x

class SpectrogramCNN(nn.Module):
    def __init__(self, hidden_dim=64, output_coordinates=1) -> None:
        super().__init__()
        self.sound_encoder = CNNEncoder(num_channels=8, hidden_dim=hidden_dim)
        self.phase_encoder = CNNEncoder(num_channels=4, hidden_dim=hidden_dim)
        self.linear = nn.Linear(2*hidden_dim, output_coordinates)
    
    def forward(self, sound, phase):
        sound = self.sound_encoder(sound)
        phase = phase.unsqueeze(0)
        phase = self.phase_encoder(phase)
        # x = torch.cat([sound, phase], dim=1)
        phase = phase.expand(sound.shape[0],-1)
        x = torch.cat([sound, phase], dim=-1)
        x = self.linear(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, stride=1, dropRate=0.0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False), 
                               nn.BatchNorm2d(out_planes)
                               ) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.bn1(x)
            out = self.conv1(F.relu(x))
        else:
            out = self.conv1(F.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out)))
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)

class WideCNNEncoder(nn.Module):
    '''Architecture inspired by Wide ResNet'''
    def __init__(self, num_channels=8, output_hidden_dimension=1024) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=64,
                               kernel_size=(3,3,2), stride=(3,3,2))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = Bottleneck(64, 128,256)
        self.layer2 = Bottleneck(256,128,256)
        self.layer3 = Bottleneck(256,128,256)
        self.layer4 = Bottleneck(256, 256, 512)
        self.layer5 = Bottleneck(512, 256, 512)
        self.layer6 = Bottleneck(512, 256, 512)
        self.layer7 = Bottleneck(512, 256, 512)
        self.layer8 = Bottleneck(512, 512, 1024)
        self.layer9 = Bottleneck(1024, 512, 1024)
        self.layer10 = Bottleneck(1024, 512, 1024)
        self.layer11 = Bottleneck(1024, 512, 1024)
        self.layer12 = Bottleneck(1024, 512, 1024)
        self.layer13 = Bottleneck(1024, 512, 1024)
        self.layer14 = Bottleneck(1024, 1024, 2048)
        self.layer15 = Bottleneck(2048, 1024, 2048)
        self.layer16 = Bottleneck(2048, 1024, 2048)
        self.linear = nn.Linear(2048, output_hidden_dimension)

    def forward(self, x):
        x = self.relu1(self.conv1(x)).squeeze(-1)
        x = self.maxpool1(x)
        for i in range(1,17):
            x = getattr(self, f"layer{i}")(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1,1)).squeeze(-1).squeeze(-1)
        x = self.linear(x)
        return x

class WideSpectrogramCNN(nn.Module):
    def __init__(self, hidden_dim=2048, num_coordinates=1) -> None:
        super().__init__()
        self.sound_encoder = WideCNNEncoder(num_channels=8, output_hidden_dimension=hidden_dim)
        self.phase_encoder = WideCNNEncoder(num_channels=4, output_hidden_dimension=hidden_dim)
        self.linear = nn.Linear(2*hidden_dim, num_coordinates)
    
    def forward(self, sound, phase):
        sound = self.sound_encoder(sound)
        phase = phase.unsqueeze(0)
        phase = self.phase_encoder(phase)
        phase = phase.expand(sound.shape[0],-1)
        x = torch.cat([sound, phase], dim=-1)
        x = self.linear(x)
        return x



class TransformerModel(nn.Module):
    '''Transformer model for audio localization.'''
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1, num_microphones: int = -1,
                 average_pool:bool = False,
                 sanity_check:bool = False):
        '''
        :arg average_pool: if to use average pooling after the application of the Transformer; pooling is done over the time dimension; alternatively the model uses the last token (default: False)
        '''

        super().__init__()
        
        self.sound_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )
        self.sanity_check = sanity_check
        if not self.sanity_check:
            self.linear = nn.Linear(input_dim+4, 1)
            self.sound_positional_encoding = PositionalEncoding(input_dim)
            self.phi_positional_encoding = PositionalEncoding(4)
            self.modulation_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(4, num_heads, hidden_dim, dropout, batch_first=True),
                num_layers
                )
            self.average_pool = average_pool
        else:
            self.linear = nn.Linear(input_dim, 1)
        
        # self.temperature = 0.01#nn.Parameter(torch.tensor(0.01))

    def forward(self, x, phi=None):
        '''
        :arg x: (batch_size, t, num_microphones) input tensor containing the recordings
        :arg phi: (1, t, 4) tensor containing the phase modulations of each rotor
        '''
        if self.sanity_check:
            #print(f"x: {x.shape} phi {phi.shape}")
            #x = torch.cat((x,phi), dim=1)
            x = softmax(x, 0.01)
            x = self.sound_transformer(torch.cat((x.double(),x),dim=1)).mean(dim=1)
            x = self.linear(x)
            return x
        if phi is None:
            phi = torch.zeros((x.shape[0], x.shape[1], 4), device=x.device)
        #x = softmax(x, 0.01)

        
        print(f"before positional encoding: {x.shape}")
        x = self.sound_positional_encoding(x)
        # create a mask over the values whose absolute value is close to zero
        
        if self.average_pool:
            x = self.sound_transformer(x).mean(dim=1)
            #x = self.sound_transformer(torch.cat((x.double(),x),dim=1)).mean(dim=1)
        else:
            # take the last token
            x = self.sound_transformer(x)[:,-1,:]

        phi = self.phi_positional_encoding(phi)
        if self.average_pool:
            phi = self.modulation_transformer(phi).mean(dim=1)
        else:
            # take the last token
            phi = self.modulation_transformer(phi)[:,-1,:]

        # x = self.transformer(torch.cat((x.double(),x),dim=1)).mean(dim=1)
        
        # Expand the phase modulation to match the batch size of the input sound  
        phi = phi.expand(x.shape[0], -1)
        # Append the phase modulation to each input sound
        x = torch.cat((x, phi), dim=-1)

        x = self.linear(x)
        return x


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(8, 64, 4, stride=4, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64,2048, 4, stride=4, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Conv1d(2048, 128, 4, stride=4, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.linear = nn.Linear(58, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        #print(f"before conv: {x.shape}")
        x = self.conv(x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        x = torch.squeeze(x, dim=-1)
        return x

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1500, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        x = x.squeeze(dim=-1)
        if x.shape[-1] >= 1500:
            x = x[:, :1500]
        else:
            x = F.pad(x, (0, 1500-x.shape[1]))
        x = self.mlp(x)
        x = torch.squeeze(x, dim=-1)
        return x

class DilatedCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(8, 64, 4, stride=4, padding=1, dilation=5),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64,2048, 4, stride=4, padding=1, dilation=5),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Conv1d(2048, 128, 4, stride=4, padding=1, dilation=5),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.linear = nn.Linear(54, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        #print(f"before conv: {x.shape}")
        x = self.conv(x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        x = torch.squeeze(x, dim=-1)
        return x

class ASTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=1, ignore_mismatched_sizes=False)
        #breakpoint()
        # self.ast.encoder.layer[-1].output = nn.Sequential(
        #     nn.Linear(3072, 1)
        # )
        #self.ast.encoder.layer[-1].output.dense = nn.Linear(3_072, 1)
        self.mlp = nn.Sequential(
            nn.GELU(),
            # nn.Linear(768, 2*768),
            # nn.ReLU(),
            # nn.Linear(2*768, 768),
            # nn.ReLU(),
            nn.Linear(768, 1),
        )
        #self.linear = nn.Linear(768, 1)
    def forward(self, x):
        x = self.ast(x).pooler_output
        #x = torch.mean(x, dim=1)
        #x = nn.functional.gelu(x)
        #print(f"before linear: {x.shape}")
        x = self.mlp(x)
        x = torch.squeeze(x, dim=-1)
        return x

class AudioModel(nn.Module):
    '''Audio model for audio localization. It embeds the recordings with a CNN and then uses a Transformer.'''
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 16,stride=16, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, 16,stride=16, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, 16,stride=16, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_dim),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )
        self.linear = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, phi=None):
        '''
        :arg x: (batch_size, num_microphones, t) input tensor containing the recordings
        :arg phi: (batch_size, 4, t) tensor containing the phase modulations of each rotor
        '''
        x = softmax(x,0.01)
        if phi is not None:
            x = torch.cat((x, phi), dim=-1)
        #x = x.permute(0, 2, 1)
        x = self.conv(x).flatten(2)
        #x = x.permute(0, 2, 1)
        x =self.transformer(torch.cat((x.double(),x),dim=1)).mean(dim=1) #self.transformer((self.transformer(torch.cat((x.double(),x),dim=1)))[:,0,:])
        #x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        x = torch.squeeze(x, dim=-1)
        return x

if __name__ == '__main__':
    batch_size, input_dim, t, hidden_dim, num_layers, num_head = 32, 8, 10, 16, 1, 1
    input_dim = 1
    num_microphones = 8
    #model = TransformerModel(input_dim, hidden_dim, num_layers, num_head, num_microphones=num_microphones)
    model = OrientationSpectrogramTransformer(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_head, num_coordinates=1)
    print(model)
    x = torch.randn(batch_size, num_microphones, t, 2)
    print(x.shape)
    phi = torch.randn(batch_size, 4, t, 2)
    print(phi.shape)
    orientation = torch.randn(batch_size, 2)
    print(orientation.shape)
    y = model(x, phi, orientation)
    #y = model(x)
    print(y.shape)