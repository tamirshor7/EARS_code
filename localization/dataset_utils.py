from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Union


scale = False
SCALE = False

class AudioLocalizationDataset(Dataset):
    """Audio localization dataset."""
    def __init__(self, path: Union[str, np.ndarray], fixed_phase: bool = False, extreme_mics: bool = False, is_using_single_microphone: bool = False, batch_first: bool = False):
        # limit_mics is used to limit the number of microphones used in the dataset
        # ATTENTION: modified code to use it in pretrained.py
        super().__init__()
        if isinstance(path, str):
            self.data = np.load(path)
            self.rec_len = self.data.shape[-1]
            if scale:
                print("Scaling the data using the min-max scaler")
                self.data[:, :, :-1] = self.data[:, :, :-1] - self.data[:, :, :-1].min(axis=2).reshape(self.data.shape[0],self.data.shape[1], 1)
                self.data[:, :, :-1] = self.data[:, :, :-1] / self.data[:, :, :-1].max(axis=2).reshape(self.data.shape[0],self.data.shape[1], 1)
            
            #self.data= torch.Tensor(np.concatenate((self.data[:,:,:-1].mean(axis=-1),np.expand_dims(self.data[:,0,-1],-1)),axis=-1))
        elif isinstance(path, np.ndarray):
            self.data = path
        else:
            raise TypeError(f"Expected path to be a string or a numpy array, got {type(path)} instead.")
        
        if SCALE:
            print("Scaling the data")
            minimum = self.data[:, :, :-1].min(axis=2).reshape(self.data.shape[0],self.data.shape[1], 1)
            maximum = self.data[:, :, :-1].max(axis=2).reshape(self.data.shape[0],self.data.shape[1], 1)
            self.data[:, :, :-1] = (self.data[:, :, :-1] - minimum) / (maximum - minimum)
            #self.data[:, :, :-1] = self.data[:, :, :-1] / self.data[:, :, :-1].max(axis=2).reshape(self.data.shape[0],self.data.shape[1], 1)

        #self.data= torch.Tensor(np.concatenate((self.data[:,:,:-1].mean(axis=-1),np.expand_dims(self.data[:,0,-1],-1)),axis=-1))
        
        
        self.fixed_phase = fixed_phase
        # self.extreme_mics = extreme_mics
        self.is_using_single_microphone = is_using_single_microphone
        # self.batch_first = batch_first
        
        # self.mean = np.mean(self.data, axis=(0,1,2))
        # self.std = np.std(self.data, axis=(0,1,2))


    def __len__(self):
        return len(self.data)

    def get_rec_len(self):
        return self.rec_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if self.extreme_mics:
        #     # choose the microphone at pi/2 and 3pi/2
        #     chooser = [1,5]
        #     sample = self.data[idx, chooser, :-1]
        # elif self.is_using_single_microphone:
        #     sample = self.data[idx, 0, :-1]
        #     sample = np.expand_dims(sample,1)
        # else:
        #     sample = self.data[idx, :, :-1]
        
        # label = self.data[idx, 0, -1]
        # label = label.astype(np.float64)

        # standardize the sample
        #sample = (sample - self.mean) / self.std
        #sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample))

        #print(f"before spectrogram: {sample.shape}")
        # convert to spectrogram
        # sample = Spectrogram(n_fft=1024, hop_length=512)(torch.from_numpy(sample))

        # convert to MFCC
        # we cut at 1500 because most samples are zero in the other samples
        #sample = MFCC(sample_rate=16000, n_mfcc=4)(torch.from_numpy(sample[..., :1500]).to(torch.float32))

        # transpose to (batch, seq_len, input_dim)
        #print(f"after spectrogram: {sample.shape}")
        #sample = sample.transpose(-1,-2).to(torch.float64)

        # if not self.is_using_single_microphone:
        #     sample = np.swapaxes(sample, -1,-2)

        # if not self.batch_first:
        #     if not self.is_using_single_microphone:
        #         sample = np.swapaxes(sample, -1,-2)
        #     else:
        #         sample = np.swapaxes(sample, -1,0)
        
        #sample = sample.astype(np.float64)
        

        if self.is_using_single_microphone:
            sample = self.data[idx, 0, :-1]
            sample = np.expand_dims(sample,0)
        else:
            sample = self.data[idx, :-1].unsqueeze(-1)

        # sample = self.data[idx, :-1].unsqueeze(-1)
        #sample = self.data[idx, :-1]




        # transpose to (batch, seq_len, input_dim)
        #sample = np.swapaxes(sample, -1,-2)
        # sample = sample.to(torch.float)
        sample = torch.as_tensor(sample, dtype=torch.double)
        label = self.data[idx, 0, -1]
        # label = label.to(torch.float)
        label = torch.as_tensor(label, dtype=torch.double)

        if self.fixed_phase:
            phase = np.zeros((sample.shape[0], 4), dtype=np.float64)
            return sample, phase, label
        return sample, label