import torch
import torch.nn as nn

class FiLMModulation(nn.Module):
    """
    Maps the condition vector to scaling (gamma) and shifting (beta)
    parameters for FiLM modulation.
    """
    def __init__(self, cond_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # For each hidden layer (except the final) output scale+shift parameters.
        self.num_mod_params = 2 * hidden_dim * (num_layers - 1)
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_mod_params)
        )

    def forward(self, cond):
        out = self.fc(cond)  # (batch, 2 * hidden_dim * (num_layers - 1))
        chunk_size = self.hidden_dim * (self.num_layers - 1)
        gamma = out[:, :chunk_size]
        beta  = out[:, chunk_size:]
        return gamma, beta

class ModulatedMLP(nn.Module):
    """
    MLP that maps 3D coordinates to aerodynamic coefficients.
    It applies FiLM (scale+shift) after each hidden layer's sine activation,
    then uses extra non-modulated layers for added expressiveness.
    Incorporates sine activations, layer normalization, and residual connections.
    """
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=256, num_layers=4, extra_layers=3, use_sine=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.extra_layers = extra_layers
        self.use_sine = use_sine

        # FiLM-modulated layers.
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.shortcut = nn.ModuleList()
        
        # First modulated layer: maps input_dim -> hidden_dim.
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        # If dimensions differ, use a projection for the residual connection.
        if input_dim != hidden_dim:
            self.shortcut.append(nn.Linear(input_dim, hidden_dim))
        else:
            self.shortcut.append(nn.Identity())
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Additional modulated layers (each with hidden_dim -> hidden_dim).
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.shortcut.append(nn.Identity())  # dimensions already match
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Extra non-modulated layers.
        self.extra = nn.ModuleList()
        self.extra_norms = nn.ModuleList()
        for _ in range(extra_layers):
            self.extra.append(nn.Linear(hidden_dim, hidden_dim))
            self.extra_norms.append(nn.LayerNorm(hidden_dim))
        
        # Final output layer.
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights for layers using sine activation if enabled.
        if self.use_sine:
            self.init_sine_weights()
        
    def init_sine_weights(self):
        # Initialize the modulated layers following guidelines from the SIREN paper.
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i == 0:
                    in_features = layer.in_features
                    bound = 1 / in_features
                    layer.weight.uniform_(-bound, bound)
                else:
                    in_features = layer.in_features
                    w0 = 30  # frequency factor (can be tuned)
                    bound = (6 / in_features) ** 0.5 / w0
                    layer.weight.uniform_(-bound, bound)
            # For extra layers, we use a standard Kaiming initialization.
            for layer in self.extra:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                
    def forward(self, coords, gamma, beta):
        chunk_size = self.hidden_dim
        h = coords
        # FiLM-modulated part.
        for i in range(self.num_layers - 1):
            h_in = h  # Save input for residual connection.
            h = self.layers[i](h)
            h = self.norms[i](h)
            # Use sine activation (or fallback to ReLU if desired).
            h = torch.sin(h) if self.use_sine else torch.relu(h)
            # FiLM modulation.
            g_i = gamma[:, i * chunk_size:(i + 1) * chunk_size]
            b_i = beta[:, i * chunk_size:(i + 1) * chunk_size]
            h = g_i * h + b_i
            # Residual connection.
            h = h + self.shortcut[i](h_in)
            
        # Extra non-modulated layers.
        for i, layer in enumerate(self.extra):
            h = layer(h)
            h = self.extra_norms[i](h)
            h = torch.sin(h) if self.use_sine else torch.relu(h)
        out = self.output_layer(h)
        return out

class FiLMNet(nn.Module):
    """
    Combines FiLMModulation and ModulatedMLP into one model.
    Takes 3D coordinates and a condition vector, then outputs predictions.
    """
    def __init__(self, cond_dim=132, coord_dim=3, output_dim=3, hidden_dim=256,
                 num_layers=4, extra_layers=3, use_sine=True):
        super().__init__()
        self.modulation_net = FiLMModulation(cond_dim, hidden_dim, num_layers)
        self.mlp = ModulatedMLP(coord_dim, output_dim, hidden_dim,
                                num_layers, extra_layers, use_sine)
        
    def forward(self, coords, cond):
        gamma, beta = self.modulation_net(cond)
        return self.mlp(coords, gamma, beta)

