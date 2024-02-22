from transformer_encoder import TransformerModel
from dataset_utils import AudioLocalizationDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import seaborn as sns
import os
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt

# from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
# import torch.nn as nn

def eval_coeff(coeff):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # dev = torch.device("cpu")

    # extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # model = nn.Sequential(
    #     AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"),
    #     nn.GELU(),
    #     nn.Linear(527, 1),
    # )
    # print('Using pretrained model')

    # load dataset
    batch_size = 4
    fixed_phase = False
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    use_clean_data = True

    data_path = f'/home/tamir.shor/EARS/data/{coeff}/inverse_problem_data_wall_x_all_0_batch8/merged_{coeff}.npy'

    dataset = AudioLocalizationDataset(data_path, fixed_phase=fixed_phase)

    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # instantiate model
    # input_dim is the embedding dimension and it needs to match the number of microphones
    if fixed_phase:
        input_dim = 12
    else:
        input_dim = 8
    hidden_dim, num_layers, num_heads = 1024, 2, 2
    model = TransformerModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
                             dropout=0)
    model.load_state_dict(torch.load("/home/tamir.shor/EARS/EARS/localization/best_model_0.2coeff_mse.pth",
                                     map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    model = model.to(dev).double()

    # instantiate optimizer

    # train

    loss_fn_mse = torch.nn.MSELoss().double()
    loss_fn_l1 = torch.nn.L1Loss().double()
    best_epoch_loss = float('inf')
    best_model_dict = None

    epoch_loss_rmse = 0
    epoch_loss_l1 = 0
    ys = None
    yhats = None
    prev_x = None
    val_loss_mse = 0
    val_loss_l1 = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
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

                if ys is None:
                    ys = y
                    yhats = y_hat
                else:
                    ys = torch.concatenate((ys,y))
                    yhats = torch.concatenate((yhats,y_hat))
                loss_mse = loss_fn_mse(y_hat, y)
                loss_l1 = loss_fn_l1(y_hat, y)
                val_loss_mse += loss_mse.item()
                val_loss_l1 += loss_l1.item()
                pbar.set_postfix({'loss mse': sqrt(loss_mse.item()), 'loss l1': loss_l1.item()})
                pbar.update(1)
    val_loss_mse = val_loss_mse / len(dataloader)
    val_loss_l1 = val_loss_l1 / len(dataloader)
    print(f'val_loss mse: {sqrt(val_loss_mse)}')
    print(f'val_loss l1: {sqrt(val_loss_l1)}')

    for i in range(len(ys)):
        if ys[i] in ys[:i]:
            ys[i] += np.random.random()/10e2

    return val_loss_mse,val_loss_l1, ys.cpu().detach().numpy(), yhats.cpu().detach().numpy()


mses = []
l1s = []
all_l1_dists = []
all_ys = []
plot_coeffs = []
coeffs = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.75]
for coeff in coeffs:
    mse_loss,l1_loss,ys,yhats = eval_coeff(coeff)
    mses.append(mse_loss)
    l1s.append(l1_loss)
    all_l1_dists += list(np.abs(yhats-ys))
    all_ys += list(p.item() for p in ys)
    plot_coeffs += [coeff]*ys.shape[0]



#plot averaged losses
# plt.plot(coeffs,mses,color='r',label="RMSE")
# plt.plot(coeffs,l1s,label = "L1")
# plt.xlabel("Absorption Coefficients")
# plt.ylabel("Error")
# plt.legend()
# plt.savefig('coeffs_losses.png')
# plt.show()

#
#plot per dist losses
data = pd.DataFrame({'Absorption Coeffs': plot_coeffs, 'GT Dists': all_ys, 'L1 Error':all_l1_dists, 'RMSE Error': np.sqrt(mses) })
data_pivoted = data.pivot('Absorption Coeffs', 'GT Dists', 'L1 Error', 'RMSE Error')
# ax = sns.heatmap(data_pivoted)
# plt.savefig('coeffs_heatmap.png')
# plt.show()

# save dataframe
data.to_csv('coeffs_data.csv')
data_pivoted.to_csv('coeffs_data_pivoted.csv')
