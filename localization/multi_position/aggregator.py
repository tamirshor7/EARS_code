from abc import ABC, abstractmethod
import torch
from sklearn.linear_model import LinearRegression
import numpy as np

class Aggregator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    @abstractmethod
    def forward(self, point_estimate, displacements):
        pass

class AverageAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, point_estimate, displacements):
        return torch.mean(point_estimate, dim=0)
class MLPAggregator(Aggregator):
    def __init__(self, number_of_points: int) -> None:
        super().__init__()
        self.number_of_points = number_of_points
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3*self.number_of_points, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )
    def forward(self, point_estimate, displacements):
        batch_size = point_estimate.shape[1]
        point_estimate = torch.transpose(point_estimate, 0,1).view(batch_size, 2*self.number_of_points)
        displacements = torch.transpose(displacements, 0, 1).view(batch_size, self.number_of_points)
        input = torch.cat((point_estimate, displacements), dim=-1)
        return self.mlp(input)
class TransformerAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=3, nhead=1),
            num_layers=2
        )
        self.linear = torch.nn.Linear(3,2)
    def forward(self, point_estimate, displacements):
        input = torch.cat((point_estimate, displacements), dim=-1)
        output = self.transformer(input).mean(dim=0)
        output = self.linear(output)
        return output
class NonDifferentiableConstantSpeedRandomWalkAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
        self.regressor = LinearRegression()
    def forward(self, point_estimate, displacements):
        spatial_coordinates = point_estimate[...,:2]
        sequence_length, batch_size = point_estimate.shape[0], point_estimate.shape[1]
        output = torch.zeros(batch_size, 3)
        for i in batch_size:
            X = np.arange(sequence_length)
            y = spatial_coordinates[:,i]
            reg = self.regressor.fit(X,y)
            output[i,:2] = reg.predict(X)
            output[i,2] = point_estimate[:,i,2]
        return output
class ConstantSpeedRandomWalkAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, point_estimate, displacements):
        sequence_length, batch_size = point_estimate.shape[0], point_estimate.shape[1]
        # use linear least square formulation: 
        X = torch.arange(sequence_length).unsqueeze(0)
        premultiplier = (torch.linalg.inv(X.T@X)@X).unsqueeze(-1)
        premultiplier = torch.tile(premultiplier, (batch_size,1,2))
        premultiplier = torch.concat(premultiplier, torch.ones(batch_size, sequence_length,1), dim=-1)
        return X@premultiplier@point_estimate 

class GeometricMedianAggregator(Aggregator):
    def __init__(self, number_of_iterations: int = 20) -> None:
        super().__init__()
        self.number_of_iterations = number_of_iterations
    @staticmethod
    def recursive_geometric_median(points, iterations=20, epsilon=1e-10):
        if iterations == 0:
            return torch.mean(points, 0, keepdim=True)
        y = GeometricMedianAggregator.recursive_geometric_median(points, iterations-1, epsilon=epsilon)
        distances = torch.linalg.norm(points-y, dim=2, keepdim=True)
        weights = 1.0 / (distances+epsilon)
        return torch.sum(weights * points, dim=0, keepdim=True)/torch.sum(weights, dim=0, keepdim=True)
    
    def forward(self, point_estimate, displacements):
        return GeometricMedianAggregator.recursive_geometric_median(point_estimate, self.number_of_iterations).squeeze(0)