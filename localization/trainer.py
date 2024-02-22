from transformer_encoder import ASTClassifier, CNN, DilatedCNN, MLP, TransformerModel
from dataset_utils import AudioLocalizationDataset
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import os

from math import sqrt

from transformers import AutoFeatureExtractor#, ASTModel
import torch.nn as nn

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#dev = torch.device("cpu")



#extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# model = nn.Sequential(
#     ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"),
#     nn.GELU(),
#     nn.Linear(527, 1),
# ).double().to(dev)

# model = ASTClassifier().double().to(dev)
# print('Using pretrained model')

# model = DilatedCNN().double().to(dev)

#model = MLP().double().to(dev)

# load dataset
batch_size = 10
fixed_phase = False
extreme_mics = False
cur_dir = os.path.dirname(os.path.abspath(__file__))
use_clean_data = True

is_using_single_microphone = False
batch_first = True

if not use_clean_data:
    data_path = os.path.join(cur_dir, '..','..','data','inverse_problem_data','inverse_problem_data.npy')
else:
    data_path = os.path.join(cur_dir, '..','..','data','inverse_problem_data','clean_inverse_problem_data.npy')

data_path = "/home/tamir.shor/EARS/data/merged.npy"
# LOCAL PATH
#data_path = os.path.join(cur_dir, '..','..','data','robustness_test', 'absorption_coefficient', '0.2', 'inverse_problem_data_wall_x_all_0_batch8','data.npy')
# SERVER PATH
#data_path = os.path.join(cur_dir, '..','..','data', '0.2', 'inverse_problem_data_wall_x_all_0_batch8','data.npy')

dataset = AudioLocalizationDataset(data_path, fixed_phase=fixed_phase, extreme_mics=extreme_mics, is_using_single_microphone=is_using_single_microphone, batch_first=batch_first)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
# train_dataset = Subset(dataset, range(0,500,250))
# val_dataset = Subset(dataset, range(10, 20))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# instantiate model
# input_dim is the embedding dimension and it needs to match the number of microphones
if extreme_mics:
    input_dim = 2
else:
    input_dim = 8
if fixed_phase:
    input_dim += 4

hidden_dim, num_layers, num_heads = 1024, 2, 1

# INPUT_DIM = 768
# NUM_MICROPHONES = 8

INPUT_DIM = 8
NUM_MICROPHONES = -1
model = TransformerModel(input_dim=INPUT_DIM, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, num_microphones=NUM_MICROPHONES,
                         dropout=0)
model = model.to(dev).double()

# instantiate optimizer
lr = 1e-3
momentum = 0.9
optimizer = Adam(model.parameters(), lr=lr)
#scheduler = ReduceLROnPlateau(optimizer, patience=3, threshold=1e-3)

# train
num_epochs = 100
loss_fn = torch.nn.MSELoss().double()
best_epoch_loss = float('inf')
best_model_dict = None

for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    epoch_loss = 0
    model.train()
    with tqdm(total=len(train_dataloader)) as pbar:
        for batch in train_dataloader:
            optimizer.zero_grad()
            # unpack batch

            if fixed_phase:
                x, phi, y = batch
            else:
                x, y = batch
            #x = x.to(dev)
            x = torch.Tensor(x).to(dev)
            if not batch_first:
                x = x.transpose(0,2)
            # x = x.squeeze()
            # x = extractor(x, sampling_rate=16_000, padding="max_length", return_tensors="pt").input_values.to(dev, dtype=torch.double)
            #print(f"after extractor: {x.shape}")
            if fixed_phase:
                phi = phi.to(dev)
            y = y.to(dev).double()
            
            #phi = torch.Tensor(phi).to(dev)
            x = torch.Tensor(x).to(dev)
            # forward pass
            if fixed_phase:
                y_hat = model(x, phi)
            else:
                y_hat = model(x)
                
            
            # compute loss
            loss = loss_fn(y_hat, y)
            # backward pass
            loss.backward()
            #breakpoint()
            # update weights
            optimizer.step()
            # update epoch loss
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': sqrt(loss.item())})
            pbar.update(1)
    epoch_loss = epoch_loss/len(train_dataloader)
    #scheduler.step(epoch_loss)
    print(f'epoch: {epoch}, loss: {sqrt(epoch_loss)}')


    # evaluate on validation set
    #model.eval()

    val_loss = 0
    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as pbar:
            for batch in val_dataloader:
                if fixed_phase:
                    x, phi, y = batch
                else:
                    x, y = batch
                x = x.to(dev)
                if fixed_phase:
                    phi = phi.to(dev)
                y = y.to(dev)
                if fixed_phase:
                    y_hat = model(x, phi)
                else:
                    y_hat = model(x)
                loss = loss_fn(y_hat, y)
                val_loss += loss.item()
                pbar.set_postfix({'loss': sqrt(loss.item())})
                pbar.update(1)
    val_loss = val_loss/len(val_dataloader)
    print(f'val_loss: {sqrt(val_loss)}')
    if val_loss < best_epoch_loss:
        best_epoch_loss = val_loss
        best_model_dict = model.state_dict()
        # save model
        model_path = os.path.join(cur_dir, 'best_model.pth')
        torch.save(best_model_dict, model_path)
        #print(f'Model saved to {model_path}')
        #print(f'Best epoch loss: {sqrt(best_epoch_loss)}.')

# save model
# model_path = os.path.join(cur_dir, 'best_model.pth')
# torch.save(best_model_dict, model_path)
# print(f'Model saved to {model_path}')
# print(f'Best epoch loss: {sqrt(best_epoch_loss)}.')

# print the predictions on the training set
model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        if fixed_phase:
            x, phi, y = batch
        else:
            x, y = batch
        x = x.to(dev)
        # x = x.squeeze()
        # x = extractor(x, sampling_rate=16_000, padding="max_length", return_tensors="pt").input_values.to(dev, dtype=torch.double)
        if fixed_phase:
            phi = phi.to(dev)
        y = y.to(dev)
        if fixed_phase:
            y_hat = model(x, phi)
        else:
            y_hat = model(x)
        print(f'y: {y}, y_hat: {y_hat}')