import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, latent_size):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(latent_size)

    def forward(self, x):
        # x shape: (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))  # (B, latent_size, N)
        # Global max pooling over points
        x = torch.max(x, dim=2, keepdim=False)[0]  # (B, latent_size)
        return x

class PointNetRegressor(nn.Module):
    def __init__(self, latent_size, output_size=9):
        super(PointNetRegressor, self).__init__()
        self.encoder = PointNetEncoder(latent_size)
        # A simple regression head
        self.regressor = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # x shape: (B, 3, N)
        latent = self.encoder(x)  # (B, latent_size)
        output = self.regressor(latent)  # (B, 9)
        return output
