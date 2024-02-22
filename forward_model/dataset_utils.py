from torch.utils.data import Dataset
from torch import load as torch_load
from numpy import load as numpy_load
import os
import re
class AudioDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        files = list(os.listdir(path))
        # this regular expression is meant to extract the first number
        # that is in the filename (it represents the id)
        reg = r"^[0-9]+"
        prog = re.compile(reg)
        # this will contain a dictionary whose keys are the ids
        # and whose values are the relevant files for each id
        self.map = {}
        # loop over the list of files to find the relevant files for each id
        for file in files:
            m = prog.match(file)
            if m:
                # if it matches, get the number which represents the id
                id = m.group(1)
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
            self.map[id] = relevant_files
    def __len__(self):
        return len(self.map)
    def __getitem__(self, index):
        '''
        :return encoder_readings, max_rec_len, omega, real_recordings, fs
        '''
        relevant_files = self.map[index]
        encoder_readings = torch_load(self.path+relevant_files['encoder'])
        other_params = numpy_load(self.path+relevant_files['other'])
        max_rec_len = other_params['max_rec_len']
        omega = other_params['omega']
        real_recordings = other_params['real_recordings']
        fs = other_params['fs']
        return encoder_readings, max_rec_len, omega, real_recordings, fs