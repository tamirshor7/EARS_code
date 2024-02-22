import os
import numpy as np
from tqdm import tqdm

def load_or_set_to_corrupted(path):
    #print(f"Checking path {path}")
    try:
        arr = np.load(path)
        if np.any(np.isnan(arr)):
            print(f"{path} has NaN values")
            os.rename(path, path.replace(".npy",".corrupted"))
            return True
        return False
    except:
        print(f"{path} is broken")
        try:
            os.rename(path, path.replace(".npy",".corrupted"))
        except:
            print(f"{path} has been removed")
            return False
        return True


sound_data_paths = [
    "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/mega_dataset/default_5.0_5.0_order_1_0.5_d_0.05/",
    "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/mega_dataset/default_5.0_5.0_order_1_0.5_d_0.05/rir/rir_indoor/", 
]
for sound_data_path in sound_data_paths:
    broken_files_counter = 0
    broken_files = []
    print(f"Processing path {sound_data_path}")
    for sound_path in tqdm(os.listdir(sound_data_path)):
        if not sound_path.endswith(".npy"):
            continue
        is_broken = load_or_set_to_corrupted(os.path.join(sound_data_path, sound_path))
        if is_broken:
            broken_files_counter += 1
            broken_files.append(os.path.join(sound_data_path, sound_path))
    print(f"These files were broken:")
    print(broken_files)
    print(f"Found {broken_files_counter} broken files")
exit(0)

get_side = lambda deformation: round(np.sqrt(deformation)*5,6) if deformation>=1 else round((np.sqrt(deformation) +((2.0*(1-np.sqrt(deformation))*0.9144+2.0*0.02)/5.0))*5.0, 6)
create_name = lambda deformation: f"rir_uniform_deformation_{get_side(deformation)}_{get_side(deformation)}_order_1_0.5_d_0.05"
    
deformations = np.arange(0.5, 2.05, 0.05)
for index, deformation in enumerate(deformations):
    print(f"Processing deformation {deformation} [{index}/{deformations.shape[0]}]")
    name = create_name(deformation=deformation)
    #rir_data_path = os.path.join("/","mnt", "walkure_public", "tamirs", "pressure_field_orientation_dataset", "robustness_test", name,"rir","rir_indoor")
    # for rir_path in tqdm(os.listdir(rir_data_path)):
    #     load_or_set_to_corrupted(os.path.join(rir_data_path, rir_path))
    