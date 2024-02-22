import torch
from torch.nn.modules import Module
from EARS.localization.multi_position import aggregator

from typing import Mapping, Any

class MultiPositionModel(torch.nn.Module):
    def __init__(self, single_position_model: torch.nn.Module, aggregator: aggregator.Aggregator) -> None:
        super().__init__()
        self.single_position_model = single_position_model
        self.aggregator = aggregator

    def forward(self, input, injected_phases=None, desired_snr_in_db:float=None, orientation=None):
        estimates = []
        displacements = []
        for i in range(input.shape[1]):
            point_estimate = self.single_position_model(input[:,i], injected_phases=injected_phases, desired_snr_in_db=desired_snr_in_db, 
                                                        orientation=orientation[:,i])
            estimates.append(point_estimate)
            # ATTENTION: torch.abs makes sense only as long as we're always rotating counter-clockwise!
            displacement = torch.abs(orientation[:,i]-orientation[:,min(i-1,0)])
            displacements.append(displacement)
        estimates = torch.stack(estimates)
        output = self.aggregator.aggregate(point_estimate=estimates, displacements=displacements)
        return output
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.single_position_model, name)

class ParallelMultiPositionModel(MultiPositionModel):
    def __init__(self, single_position_model: torch.nn.Module, aggregator: aggregator.Aggregator) -> None:
        super().__init__(single_position_model=single_position_model, aggregator=aggregator)
    def forward(self, input, injected_phases=None, desired_snr_in_db:float=None, orientation=None, system_noise_gain:float=0.1):
        # print(f"[debug] input to multi position shape {input.shape}")
        # print(f"[debug] single position model is {self.single_position_model}")
        batch_size, number_of_positions = input.shape[0], input.shape[1]
        input = input.view(batch_size*number_of_positions, input.shape[2], input.shape[3], input.shape[4])

        # ATTENTION: this treatment of displacements makes sense only as long as we're always rotating counter-clockwise!
        displacements = torch.abs(torch.diff(orientation))
        displacements = torch.cat((torch.zeros(batch_size,1, device=displacements.device), displacements), dim=-1)
        displacements = torch.transpose(displacements, 0,1).unsqueeze(-1)

        orientation = orientation.view(batch_size*number_of_positions)
        estimates = self.single_position_model(input, injected_phases=injected_phases, desired_snr_in_db=desired_snr_in_db, 
                                                        orientation=orientation, system_noise_gain=system_noise_gain)
        estimates = torch.transpose(estimates.view(batch_size, number_of_positions, -1), 0, 1)
        output = self.aggregator(estimates, displacements)
        return output
    @staticmethod
    def is_aggregator_state_dict(state_dict: Mapping[str, Any]):
        return any(k.startswith('single_position_model') for k in state_dict.keys())
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if ParallelMultiPositionModel.is_aggregator_state_dict(state_dict):
            super().load_state_dict(state_dict, strict)
        else:
            self.single_position_model.load_state_dict(state_dict)
    
class DeepRobustMultiPositionModel(torch.nn.Module):
    def __init__(self, single_position_model: torch.nn.Module, 
                 dim_feedforward:int=2_048, num_heads:int = 8, dropout: float = 0.2, num_layers:int = 3,
                 device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        super().__init__()
        # ATTENTION: if you change the name of the variable self.single_position_model
        #            you need to also change the string used in 
        #            DeepRobustMultiPositionModel.is_aggregator_state_dict !
        self.single_position_model = single_position_model.to(device)
        hidden_dimension_single_position_model: int = self.single_position_model.backward_model.linear.in_features
        num_coordinates: int = self.single_position_model.backward_model.linear.out_features
        dtype = self.single_position_model.backward_model.linear.weight.dtype
        self.single_position_model.backward_model.linear = torch.nn.Identity()

        # ATTENTION: if you change the name of self.robust_aggregator you need to change it also in
        #            train_pipeline/optimizer_load_state_dict and in train_separate/optimizer_load_state_dict
        self.robust_aggregator = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_dimension_single_position_model, num_heads, dim_feedforward,
                                       dropout=dropout, batch_first=True, device=device, dtype=dtype),
            num_layers
        )
        # ATTENTION: if you change the name of self.linear you need to change it also in
        #            train_pipeline/optimizer_load_state_dict and in train_separate/optimizer_load_state_dict
        self.linear = torch.nn.Linear(hidden_dimension_single_position_model, num_coordinates, device=device, dtype=dtype)

    def forward(self, input, injected_phases=None, desired_snr_in_db:float=None, orientation=None, system_noise_gain=None):
        # input is expected to have shape (batch_size, num_angles, microphones, stft_dim_1, stft_dim_2, magnitude_phase i.e. 2)
        # injected_phases is expected to have shape (num_rotors i.e. 4, stft_dim_1, stft_dim_2, magnitude_phase i.e. 2)
        # orientation is expected to have shape (batch_size, num_angles, cos_sin i.e. 2)
        batch_size, num_angles = input.shape[0], input.shape[1]
        input = input.view([batch_size*num_angles]+[*input.shape[2:]])
        orientation = orientation.view([batch_size*num_angles]+[*orientation.shape[2:]])

        encoded_estimates = self.single_position_model(input, injected_phases=injected_phases, desired_snr_in_db=desired_snr_in_db, 
                                                        orientation=orientation, system_noise_gain=system_noise_gain)
        encoded_estimates = encoded_estimates.view([batch_size, num_angles]+[*encoded_estimates.shape[1:]])

        encoded_estimates = self.robust_aggregator(encoded_estimates)

        encoded_estimates = encoded_estimates.mean(dim=1)

        output = self.linear(encoded_estimates)
        return output
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.single_position_model, name)
    
    @staticmethod
    def is_aggregator_state_dict(state_dict: Mapping[str, Any]):
        return any(k.startswith('single_position_model') for k in state_dict.keys())
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        #return super().load_state_dict(state_dict, strict)
        if DeepRobustMultiPositionModel.is_aggregator_state_dict(state_dict):
            #print(f"[debug] loading complete state dictionary")
            super().load_state_dict(state_dict, strict)
        else:
            #print(f"[debug] loading only point model state dictionary")
            # remove the parameters that we won't be using
            del state_dict["backward_model.linear.weight"]
            del state_dict["backward_model.linear.bias"]
            self.single_position_model.load_state_dict(state_dict)
            # need to set linear model to identity since the loaded model should have an actual linear layer!
            self.single_position_model.linear = torch.nn.Identity()