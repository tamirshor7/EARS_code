from torch.utils.data import DataLoader
from EARS.localization.phase_modulation.modulation_dataset import ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient
from tqdm import tqdm
from EARS.localization.phase_modulation.rir import Rir

def generate_all_instances_dataset(absorption_coefficient=0.2, step_distance=0.0008):
    dataset = ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient(absorption_coefficient=absorption_coefficient, step_distance=step_distance)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # out is used to make sure that due to optimization the rir is actually calculated and stored
    out = None
    with tqdm(total=len(dataset)) as pbar:
        for input, _ in dataloader:
            del out
            absorption_coefficients = [i[0] for i in input]
            distances_from_wall = [i[1] for i in input]
            rir = Rir(absorption_coefficient=absorption_coefficients,
                        distance_from_wall=distances_from_wall)
            out = rir._get_rir()
            pbar.update(1)
            pbar.set_postfix({'absorption_coefficient': absorption_coefficients[0],
                                'distance_from_wall': distances_from_wall[0]})
    print("Done generating all instances of the dataset.")

if __name__ == "__main__":
    # take absorption_coefficient and step_distance as arguments from command line
    import sys
    if len(sys.argv) == 3:
        absorption_coefficient = float(sys.argv[1])
        step_distance = float(sys.argv[2])
        generate_all_instances_dataset(absorption_coefficient=absorption_coefficient, step_distance=step_distance)
    else:
        print("Usage: python generate_all_instances_dataset.py absorption_coefficient step_distance")
            