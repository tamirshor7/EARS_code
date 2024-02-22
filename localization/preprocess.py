import torch 
import torchaudio.transforms as transforms
from math import prod

def preprocess_audio_waveform(data, sample_rate=None, desired_sample_rate=None, convert_to_db:bool=False, standardize:bool=False, n_fft:int=40, hop_length:int=40, window_length:int=20):

    # Resample if needed
    if sample_rate is not None and desired_sample_rate is not None:
        raise NotImplementedError("Resampling is not implemented yet")
        sample_rates = torch.tensor([waveform.shape[1] for waveform in waveforms])
        resample_indices = sample_rates != desired_sample_rate
        resampler = transforms.Resample(orig_freq=sample_rates[resample_indices], new_freq=desired_sample_rate)
        waveforms[resample_indices] = resampler(waveforms[resample_indices])

    # Convert mono audio to stereo if needed
    # mono_indices = waveforms.size(1) == 1
    # waveforms = waveforms.repeat(1, 2, 1)[mono_indices]

    # Apply Short-Time Fourier Transform (STFT)
    # n_fft = n_fft  # Set your desired number of FFT points
    # hop_length = hop_length  # Set your desired hop length
    # stft = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    # spectrograms = stft(data)
    # convert data to 2D
    original_shape = data.shape
    new_shape = torch.Size((prod(original_shape[:-1]),original_shape[-1]))
    data = data.view(new_shape).to(torch.float64)
    
    output = torch.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=window_length, onesided=True, return_complex=True, center=False,
                        window=torch.hann_window(window_length).to(data.device))
    # convert back to its original shape
    output = output.view([*original_shape[:-1]]+[*output.shape[-2:]])
    # separate magnitude and phase
    output = torch.stack([torch.abs(output), torch.angle(output)], dim=-1)
    output = output.to(torch.float64)

    if len(output.shape) > 5:
        output= output.flatten(start_dim=0, end_dim=-5)


    if convert_to_db:
        # Convert to decibel scale
        top_db = 80  # Set the threshold for the dB scaling
        output = transforms.AmplitudeToDB(top_db=top_db)(output)

    if standardize:
        # Standardize spectrograms
        output = torch.nn.functional.normalize(output, p=2, dim=-1)
        # means = output.mean(dim=-1, keepdim=True)
        # stds = output.std(dim=-1, keepdim=True)
        # output = (output - means) / (stds + 1e-10)

    return output

def preprocess(data, scale=True, reshape:bool = True, average:bool = True, permute:bool = True, epsilon:float = 1e-3, add_epsilon:bool = False):
    if scale:
        # apply min-max scaler
        minimum, _ = torch.min(data, dim=-1, keepdim=True)
        maximum, _ = torch.max(data, dim=-1, keepdim=True)
        epsilon = epsilon if add_epsilon else 0
        data = (data-minimum)/(maximum-minimum+epsilon)
    if reshape:
        # reshape to have 3 dimensions
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        elif len(data.shape) == 4:
            data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])
    if average:
        # average over time
        data = torch.mean(data, dim=-1, keepdim=True)
    if permute:
        # permute to (batch_size, n_samples, n_mics)
        data = data.permute(0,2,1)
    return data