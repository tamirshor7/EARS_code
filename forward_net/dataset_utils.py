from torch.utils.data import Dataset
from torch import load as torch_load
from numpy import load as numpy_load
import os
import re
from torch import tensor, stack, unsqueeze, Tensor, float64, float32
from torch.nn.functional import pad
from constants import PAD_VALUE
from typing import Callable

class AudioDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        files = list(os.listdir(path))
        # this regular expression is meant to extract the first number
        # that is in the filename (it represents the id)
        reg = r"^[0-9]+"
        prog = re.compile(reg)
        #this regular expression is meant to extract the recording mode from the file
        # (CW or CCW)
        reg_recording_mode = r"recmodeC(W|CW)"
        prog_recording_mode = re.compile(reg_recording_mode)
        # this will contain a dictionary whose keys are the ids
        # and whose values are the relevant files for each id
        self.map = {}
        # loop over the list of files to find the relevant files for each id
        for file in files:
            m = prog.match(file)
            if m:
                # if it matches, get the number which represents the id
                id = m.group(0)
            else:
                # else ignore it (the name of the file is not correct)
                continue
            # get the list of relevant files for this id and add this new file
            relevant_files = self.map.get(id, {})
            if file.endswith('_encoder.pt'):
                relevant_files['encoder'] = file
            elif file.endswith('.npz'):
                relevant_files['other'] = file
            else:
                continue
            # save the recording mode

            # search checks the regex from every position of the string (match checks just from the beginning)
            m = prog_recording_mode.search(file)
            if m:
                relevant_files['recording_mode'] = 1 if m.group(0).strip('recmode') == 'CW' else 0
            else:
                continue
            self.map[id] = relevant_files
    def __len__(self):
        return len(self.map)
    def __getitem__(self, index):
        '''
        :return encoder_readings, max_rec_len, omega, real_recordings, fs
        '''
        relevant_files = self.map[str(index)]
        encoder_readings = torch_load(self.path+relevant_files['encoder'])
        other_params = numpy_load(self.path+relevant_files['other'],allow_pickle=True) #Weird crash if not added

        return {
            "encoder_readings":encoder_readings ,
            "max_rec_len": int(other_params['max_rec_len']),
            "omega": float(other_params['omega']),
            "real_recordings": other_params['real_recordings'],
            "fs": float(other_params['fs']),
            "recording_mode": relevant_files['recording_mode']
        }

class SingleAudioDataset(Dataset):
    def __init__(self, path, requested_id:int, set_len:int = -1) -> None:
        '''Reduce the dataset to a single use-case'''
        super().__init__()
        self.path = path
        self.requested_id = requested_id
        self.set_len = set_len
        files = list(os.listdir(path))
        # this regular expression is meant to extract the first number
        # that is in the filename (it represents the id)
        reg = r"^[0-9]+"
        prog = re.compile(reg)
        #this regular expression is meant to extract the recording mode from the file
        # (CW or CCW)
        reg_recording_mode = r"recmodeC(W|CW)"
        prog_recording_mode = re.compile(reg_recording_mode)
        # this will contain a dictionary whose keys are the ids
        # and whose values are the relevant files for each id
        self.map = {}
        # loop over the list of files to find the relevant files for each id
        for file in files:
            m = prog.match(file)
            if m:
                # if it matches, get the number which represents the id
                id = m.group(0)
            else:
                # else ignore it (the name of the file is not correct)
                continue
            # filter all of the other files!
            if id != str(requested_id):
                continue
            # get the list of relevant files for this id and add this new file
            relevant_files = self.map.get(id, {})
            if file.endswith('_encoder.pt'):
                relevant_files['encoder'] = file
            elif file.endswith('.npz'):
                relevant_files['other'] = file
            else:
                continue
            # save the recording mode

            # search checks the regex from every position of the string (match checks just from the beginning)
            m = prog_recording_mode.search(file)
            if m:
                relevant_files['recording_mode'] = 1 if m.group(0).strip('recmode') == 'CW' else 0
            else:
                continue
            self.map[id] = relevant_files
    def __len__(self):
        return len(self.map) if self.set_len == -1 else self.set_len
    def __getitem__(self, index):
        '''
        :return encoder_readings, max_rec_len, omega, real_recordings, fs
        '''
        index = self.requested_id
        relevant_files = self.map[str(index)]
        encoder_readings = torch_load(self.path+relevant_files['encoder'])
        other_params = numpy_load(self.path+relevant_files['other'],allow_pickle=True) #Weird crash if not added

        return {
            "encoder_readings":encoder_readings ,
            "max_rec_len": int(other_params['max_rec_len']),
            "omega": float(other_params['omega']),
            "real_recordings": other_params['real_recordings'],
            "fs": float(other_params['fs']),
            "recording_mode": relevant_files['recording_mode']
        }

# helper methods for DataLoader
def pad_sequence(data, padding_value = PAD_VALUE):
    '''
    Pads a list of tensors along the last dimension so that they all have the maximum length
    '''
    max_len = max([i.shape[-1] for i in data])
    # shall we change it to stack?
    # return cat([pad(i, (0, max_len -i.shape[-1]), mode='constant', value=padding_value) for i in data])
    return stack([pad(i, (0, max_len -i.shape[-1]), mode='constant', value=padding_value) for i in data])

def collate_fn(data):
    encoder_readings = []
    max_rec_len = []
    omega = []
    real_recordings = []
    fs = []
    recording_mode = []
    for d in data:
        encoder_readings.append(d['encoder_readings'])
        max_rec_len.append(d['max_rec_len'])
        omega.append(d['omega'])
        real_recordings.append(tensor(d['real_recordings']))
        fs.append(d['fs'])
        recording_mode.append(d['recording_mode'])
    encoder_readings = pad_sequence(encoder_readings, padding_value=PAD_VALUE)
    max_rec_len = tensor(max_rec_len)
    omega = tensor(omega)
    real_recordings = pad_sequence(real_recordings, padding_value=PAD_VALUE)
    fs = tensor(fs)
    recording_mode = tensor(recording_mode)
    return {
            "encoder_readings":encoder_readings ,
            "max_rec_len": max_rec_len,
            "omega": omega,
            "real_recordings": real_recordings,
            "fs": fs,
            "recording_mode" : recording_mode
        }

def pad_sequence_gru(data, padding_value = PAD_VALUE):
    '''
    Pads a list of tensors along the last dimension so that they all have the maximum length
    '''
    max_len = max([i.shape[-1] for i in data])
    data = sorted(data, key=lambda x: x.shape[-1], reverse=True)
    src_len = [i.shape[-1] for i in data]
    # transpose it so that we can use it with pack_padded_sequence 
    # (it needs shape (batch, sequence, other) currently it's (batch, other, sequence))
    #transform = lambda x: transpose(pad(x, (0, max_len-x.shape[-1]), mode='constant', value=padding_value), 1,0)

    # We unsqueeze it so that the shape of every data point will be (sequence, 1)
    # In this way out will have the shape (batch, sequence, 1) (this is required by GRU)
    transform = lambda x: unsqueeze(pad(x, (0, max_len-x.shape[-1]), mode='constant', value=padding_value), -1)
    #out = transpose(cat([pad(i, (0, max_len -i.shape[-1]), mode='constant', value=padding_value) for i in data]), 1,0)
    out = stack([transform(i) for i in data])
    return out, src_len

def collate_fn_gru(data):
    encoder_readings = []
    max_rec_len = []
    omega = []
    real_recordings = []
    fs = []
    recording_mode = []
    for d in data:
        encoder_readings.append(d['encoder_readings'])
        max_rec_len.append(d['max_rec_len'])
        omega.append(d['omega'])
        real_recordings.append(tensor(d['real_recordings'].astype('float32'))) #convert to type otherwise np object types crahse
        fs.append(d['fs'])
        recording_mode.append(d['recording_mode'])
    encoder_readings, src_len = pad_sequence_gru(encoder_readings, padding_value=PAD_VALUE)
    max_rec_len = tensor(max_rec_len)
    omega = tensor(omega)
    real_recordings = pad_sequence(real_recordings, padding_value=PAD_VALUE).to(float64)
    fs = tensor(fs)
    # we unsqueeze recording_mode to concatenate it with other_input later
    recording_mode = unsqueeze(tensor(recording_mode), dim=-1)
    return {
            "encoder_readings":encoder_readings ,
            "max_rec_len": max_rec_len,
            "omega": omega,
            "real_recordings": real_recordings,
            "fs": fs,
            "recording_mode" : recording_mode,
            "src_len": src_len
        }
def collate_fn_gru_all_dist(data):
    encoder_readings = []
    max_rec_len = []
    omega = []
    real_recordings = []
    fs = []
    recording_mode = []
    for d in data:
        encoder_readings.append(d['encoder_readings'][0].to(float32))
        max_rec_len.append(d['max_rec_len'])
        omega.append(d['omega'])
        real_recordings.append(tensor(d['real_recordings'].astype('float32'))) #convert to type otherwise np object types crahse
        fs.append(d['fs'])
        recording_mode.append(d['recording_mode'])
    encoder_readings, src_len = pad_sequence_gru(encoder_readings, padding_value=PAD_VALUE)
    max_rec_len = tensor(max_rec_len)
    omega = tensor(omega)
    real_recordings = pad_sequence(real_recordings, padding_value=PAD_VALUE).to(float64)
    fs = tensor(fs)
    # we unsqueeze recording_mode to concatenate it with other_input later
    recording_mode = unsqueeze(tensor(recording_mode), dim=-1)
    return {
            "encoder_readings":encoder_readings ,
            "max_rec_len": max_rec_len,
            "omega": omega,
            "real_recordings": real_recordings,
            "fs": fs,
            "recording_mode" : recording_mode,
            "src_len": src_len
        }
def collate_device(collate_fn: Callable, device: str) -> Callable:
    '''
    Returns a collate function that moves the data to the device
    :param collate_fn: the collate function to wrap
    :param device: the device to move the data to
    '''
    def collate_fn_device(batch):
        return tuple([val.to(device) if type(val) == Tensor else val for val in collate_fn(batch).values()])
    return collate_fn_device