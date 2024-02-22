import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size:int = 1024, num_layers:int =3, ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    def forward(self, x, phases=None):
        x, _ = self.gru(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        #x = torch.squeeze(x, dim=-1)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, seq_len, input_size, hidden_size, num_layers = 10, 1000, 8, 1024, 3
    x = torch.rand(batch_size, seq_len, input_size, device=device)
    model = GRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    y = model(x)
    print(y.shape)