from transformers import ASTFeatureExtractor, ASTModel
import torch.nn as nn
import torch
import os
from dataset_utils import AudioLocalizationDataset
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from transformer_encoder import ASTClassifier
from math import sqrt
# extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
#processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# specify mean and std of the training set
# specify or change sampling rate
sampling_rate = 16_000
feature_extractor = ASTFeatureExtractor(sampling_rate=sampling_rate)
print(feature_extractor)
# print(dir(feature_extractor))
# print(feature_extractor._extract_fbank_features)
# print(type(feature_extractor._extract_fbank_features))

# exit()

# model = nn.Sequential(
#     ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"),
#     nn.GELU(),
#     nn.Linear(768, 1),
# )

model = ASTClassifier()
if torch.__version__ >= '2.0':
    model = torch.compile(model, mode='max-autotune')
# model = ST(hidden_dim=64, num_layers=2, num_heads=1, dropout=0.0)
print(model)
#breakpoint()

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(dev).double()
model.train()

# load dataset
batch_size = 10
print(f"batch size: {batch_size}")
fixed_phase = False
extreme_mics = False
is_using_single_microphone = True
cur_dir = os.path.dirname(os.path.abspath(__file__))
use_clean_data = True
if not use_clean_data:
    data_path = os.path.join(cur_dir, '..','..','data','inverse_problem_data','inverse_problem_data.npy')
else:
    data_path = os.path.join(cur_dir, '..','..','data','inverse_problem_data','clean_inverse_problem_data.npy')
#dataset = Subset(AudioLocalizationDataset(data_path, fixed_phase=fixed_phase, extreme_mics=extreme_mics, is_using_single_microphone=is_using_single_microphone), range(4))
dataset = AudioLocalizationDataset(data_path, fixed_phase=fixed_phase, extreme_mics=extreme_mics, is_using_single_microphone=is_using_single_microphone)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# instantiate optimizer
lr = 5e-6
optimizer = Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5], gamma=0.5)

num_epochs = 100
loss_fn = nn.MSELoss().double()

with tqdm(total=num_epochs) as pbar:
    for epoch in range(num_epochs):
        epoch_loss = 0
        with tqdm(total=len(dataloader)) as pbar2:
            for batch in dataloader:
                optimizer.zero_grad()
                x_batch, y = batch
                x_batch = x_batch.squeeze()
                #print(x_batch.shape)
                # print(y.shape)
                #x = extractor(x, return_tensors='pt').input_values.to(dev)
                #x = processor(x, return_tensors='pt').input_values.to(dev)

                # x = feature_extractor(x, sampling_rate=sampling_rate, padding="max_length", return_tensor="pt").input_values
                # #.to(dev)
                # x = torch.as_tensor(np.asarray(x), dtype=torch.float64).to(dev)

                if batch_size > 1:
                    x_batch = [torch.as_tensor(feature_extractor(x, sampling_rate=sampling_rate, padding="max_length", return_tensor="pt").input_values[0], dtype=torch.double, device=dev) for x in x_batch]
                    x_batch = torch.stack(x_batch).to(dev)
                else:
                    x_batch = torch.as_tensor(feature_extractor(x_batch, sampling_rate=sampling_rate, padding="max_length", return_tensor="pt").input_values[0], dtype=torch.double, device=dev).unsqueeze(0)   
                # print(f"after feature extractor: {x}")
                # print(f"after feature extractor: {x.shape}")
                y = y.to(dev)
                y_hat = model(x_batch)
                loss = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                pbar2.set_postfix({'loss': sqrt(loss.item())})
                pbar2.update(1)
                epoch_loss += loss.item()
                #print(f"loss: {sqrt(loss.item())}")
        epoch_loss /= len(dataloader)
        pbar.set_postfix({'loss': sqrt(epoch_loss)})
        pbar.update(1)
        scheduler.step()