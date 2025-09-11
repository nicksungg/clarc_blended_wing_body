import torch
import torch.nn as nn

class FiLMModulation(nn.Module):
    """
    Maps the condition vector to scaling (gamma) and shifting (beta)
    parameters for FiLM modulation in the MLP.
    """
    def __init__(self, cond_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # For each hidden layer (except the final) we output scale+shift parameters.
        self.num_mod_params = 2 * hidden_dim * (num_layers - 1)
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_mod_params)
        )

    def forward(self, cond):
        out = self.fc(cond)  # (batch, 2*hidden_dim*(num_layers-1))
        chunk_size = self.hidden_dim * (self.num_layers - 1)
        gamma = out[:, :chunk_size]
        beta  = out[:, chunk_size:]
        return gamma, beta


class ModulatedMLP(nn.Module):
    """
    MLP that maps 3D coordinates to aerodynamic coefficients.
    Applies FiLM (scale+shift) after each hidden layer's ReLU,
    then processes the output through extra (non-modulated) layers
    to increase the model's expressiveness.
    """
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=256, num_layers=4, extra_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.extra_layers = extra_layers

        # Build the FiLM-modulated layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Extra non-modulated layers to boost expressiveness
        self.extra = nn.ModuleList()
        for _ in range(extra_layers):
            self.extra.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, coords, gamma, beta):
        chunk_size = self.hidden_dim
        h = coords
        # FiLM-modulated part
        for i in range(self.num_layers - 1):
            h = self.layers[i](h)
            h = torch.relu(h)
            # Apply FiLM modulation
            g_i = gamma[:, i * chunk_size:(i + 1) * chunk_size]
            b_i = beta[:, i * chunk_size:(i + 1) * chunk_size]
            h = g_i * h + b_i
        
        # Extra non-modulated layers
        for layer in self.extra:
            h = torch.relu(layer(h))
        
        out = self.output_layer(h)
        return out


class FiLMNet(nn.Module):
    """
    Combines FiLMModulation and ModulatedMLP into one model.
    Takes in 3D coordinates and a condition vector and outputs predictions.
    """
    def __init__(self, cond_dim=14, coord_dim=3, output_dim=3, hidden_dim=256, num_layers=4, extra_layers=2):
        super().__init__()
        self.modulation_net = FiLMModulation(cond_dim, hidden_dim, num_layers)
        self.mlp = ModulatedMLP(coord_dim, output_dim, hidden_dim, num_layers, extra_layers)

    def forward(self, coords, cond):
        gamma, beta = self.modulation_net(cond)
        return self.mlp(coords, gamma, beta)
