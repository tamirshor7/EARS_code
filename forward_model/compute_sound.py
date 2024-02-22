import torch
import numpy as np
import os
import argparse
# import asyncio
# import aiofiles
from EARS.io import hdf5
from multiprocessing import Pool
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rir_path', required=True, type=str, help='Path to the directory that contains the RIRs')
    parser.add_argument('-batch_size', type=int, default=300, help='Batch size')
    parser.add_argument('-chunk', default=None, type=int, help="Which chunk to work on")
    parser.add_argument('-total_gpus_available', default=None, type=int, help="Store the available amount of gpus")
    parser.add_argument('-gpu', required=True, type=str, help="Which GPU to use")
    parser.add_argument('-use_npy', default=False, action='store_true', help="Whether to use .npy files")
    args = parser.parse_args()
    return vars(args)

def check_arguments(args):
    assert os.path.exists(args['rir_path']), f"The path to the RIRs {args['rir_path']} does not exist"
    assert (args['chunk'] is not None and args['total_gpus_available'] is not None) or (args['chunk'] is None and args['total_gpus_available'] is None), f"You specified one among chunk and total_gpus_available but not both of them (got chunk {args['chunk']} {args['total_gpus_available']}). Please either specify both of them or none of them"
    if args['chunk'] is not None:
        assert 0<=args["chunk"]<=args["total_gpus_available"], \
            f"Got args['chunk'] {args['chunk']} and args['total_gpus_available'] {args['total_gpus_available']}, please set them properly"

@torch.jit.script
def convolve(rir:torch.Tensor, signals:torch.Tensor):
    reshaped_rir = rir.reshape(rir.shape[0]*rir.shape[1],rir.shape[2], rir.shape[3])
    output = torch.zeros(rir.shape[0]*rir.shape[1],4, signals.shape[-1]-reshaped_rir.shape[-1]+1, dtype=torch.float64, device=rir.device)
    for i in range(4):
        output[:,i] = torch.nn.functional.conv1d(signals[:,256*i:256*(i+1)], torch.flip(reshaped_rir[:,256*i:256*(i+1)], [-1])).squeeze()
    output = output.reshape(rir.shape[0], rir.shape[1], 4, output.shape[-1])
    output = torch.permute(output, (0,2,1,3))
    return output

def filter_files(data_path, extension=".hdf5"):
    # filter according to extension
    extension_filter = lambda x: x.endswith(extension)
    # check if already computed
    already_computed = set(filter(extension_filter, os.listdir(os.path.join(data_path, "..", ".."))))

    files = set(os.listdir(data_path))
    files = set(filter(extension_filter, files))
    files = files.difference(already_computed)

    files = list(files)

    return files

class RirDataset(torch.utils.data.Dataset):
    def __init__(self, data_path:str, chunk:int = None, total_gpus_available:int = None,
                 file_extension=".npy") -> None:
        assert data_path is not None and os.path.exists(data_path), f"Invalid data_path: got {data_path}"
        super().__init__()
        self.data_path = data_path
        self.file_extension = file_extension
        self.data = filter_files(self.data_path, self.file_extension)
        print(f"About to compute {len(self.data)} files")
        if chunk is not None:
            start_index = int((len(self.data)/total_gpus_available)*chunk)
            end_index = len(self.data) if (chunk==(total_gpus_available-1)) else int((len(self.data)/total_gpus_available)*(chunk+1))
            rirs_path = rirs_path[start_index:end_index]

    def __len__(self) -> int:
        return len(self.data)
    
    def get_data(self, complete_path):
        try:
            data = torch.from_numpy(np.load(complete_path))
        except:
            raise IOError(f"Cannot read from {complete_path}")
        return data
    
    def __getitem__(self, index:int) -> torch.Tensor:
        path = self.data[index]
        complete_path = os.path.join(self.data_path, path)
        data = self.get_data(complete_path)
        return data, path

class Hdf5RirDataset(RirDataset):
    def __init__(self, data_path: str, chunk: int = None, total_gpus_available: int = None) -> None:
        super().__init__(data_path, chunk, total_gpus_available, file_extension='.hdf5')
    def get_data(self, complete_path):
        try:
            data = hdf5.load_torch(complete_path)
        except:
            raise IOError(f"Cannot read from {complete_path}")
        return data
    
def collate_fn(batch):
    max_len = max([item[0].shape[-1] for item in batch])
    rirs = [torch.nn.functional.pad(item[0], (0, max_len - item[0].shape[-1])) for item in batch]
    names = [item[1] for item in batch]
    return torch.stack(rirs), names

def get_dataloader(rirs_path:str, batch_size:int, use_hdf5=True):
    if use_hdf5:
        dataset = Hdf5RirDataset(rirs_path)
    else:
        dataset = RirDataset(rirs_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False,collate_fn=collate_fn)
    return dataloader

# async def save_tensor(sound,save_path, name):
#     sound = sound.cpu().numpy()
#     #np.save(os.path.join(save_path, name), sound)
#     async with aiofiles.open(os.path.join(save_path, name), 'wb') as file:
#         await file.write(sound.tobytes())

def save_tensor_hdf5(data_input):
    path, tensor = data_input
    hdf5.save(path, tensor)

def save_tensor_numpy(data_input):
    path, tensor = data_input
    np.save(path, tensor)

#async def main():
def main():
    args = parse_arguments()
    check_arguments(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']

    rirs_path: str = args['rir_path']
    save_path: str = os.path.join(rirs_path, "..", "..")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise ValueError("GPU is not accessible")

    SIGNALS_PATH: str = "/home/gabriele/EARS_project/EARS/localization/robustness_tests/signals.npy"
    SIGNALS: torch.Tensor = torch.from_numpy(np.load(SIGNALS_PATH)).to(device, dtype=torch.float64).unsqueeze(0)

    dataloader = get_dataloader(rirs_path, args['batch_size'], use_hdf5=not args['use_npy'])

    #all_tasks = []

    cpu_pool = os.cpu_count()-1

    for rirs, names in tqdm(dataloader):
        rirs = rirs.to(device)
        sounds = convolve(rirs, SIGNALS)
        # for index, name in enumerate(names):
        #     sound = sounds[index].cpu().numpy()
        #     np.save(os.path.join(save_path, name), sound)

        data_inputs = [(os.path.join(save_path, name), sounds[index].cpu().numpy()) for index, name in enumerate(names)]
        if args['use_npy']:
            with Pool(cpu_pool) as pool:
                pool.map(save_tensor_numpy, data_inputs)
        else:
            with Pool(cpu_pool) as pool:
                pool.map(save_tensor_hdf5, data_inputs)
        
        # for index, name in enumerate(names):
        #     asyncio.run(save_tensor(sounds[index], save_path, name))
        # tasks = [asyncio.create_task(save_tensor(sounds[index], save_path, name)) for index, name in enumerate(names)]
        # all_tasks.extend(tasks)
    # done, pending = await asyncio.wait(all_tasks)
    #await asyncio.wait(all_tasks)
    

if __name__ == '__main__':
    #asyncio.run(main())
    main()