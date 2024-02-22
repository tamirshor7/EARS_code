import webdataset as wds
import torch
import h5py
import io
from itertools import islice

def load_torch(bytes):
    stream = io.BytesIO(bytes)
    with h5py.File(stream, "r") as f:
        torch_tensor = torch.as_tensor(f['data'][...])
    return torch_tensor
def filename_to_coordinates(filename:str):
    filename = filename.removeprefix('./')
    raw_coordinates = filename.split('_')
    coordinates = torch.tensor(tuple(map(lambda x: float(x)/10**8, raw_coordinates)))
    return coordinates
def collate_fn(data):
    print(f"got data {data}")
    coordinates = data[0]
    sound = data[1]
    print(f"got sound {sound} {type(sound)} {sound.shape}", flush=True)
    max_len: int = max(sound, key=lambda x:x.shape[-1])
    sound = torch.stack(list(map(lambda x: torch.nn.functional.pad(x, (0, max_len - x.shape[-1])), sound)))
    return sound, coordinates

def build_dataloader(path: str, batch_size: int):
    dataset = (wds.WebDataset(path)\
               .shuffle(batch_size)\
                .rename(filename="__key__", bytes="hdf5")\
                .map_dict(filename=filename_to_coordinates, bytes=load_torch)\
                .to_tuple("filename", "bytes")
                )
    dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size),
                                             num_workers=1, batch_size=None,
                                             collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    url: str = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/dataset.tar"
    batch_size: int = 2
    dataloader = build_dataloader(url, batch_size)
    for idx, i in enumerate(islice(dataloader, 0,3,1)):
        print(f"{idx}: {i}")