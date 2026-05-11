import torch.nn as nn
 
class LinearModel(nn.Module):
    """Single linear layer with dropout.
    Baseline model : no hidden layer, no non-linearity."""
 
    def __init__(self, n_in, n_out, dropout=0.4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_in, n_out)
        )
 
    def forward(self, x):
        return self.model(x)
    
class OneHiddenLayer(nn.Module):
    """One hidden layer with ReLU activation and dropout."""
 
    def __init__(self, n_in, n_out, hidden, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, n_out)
        )
 
    def forward(self, x):
        return self.model(x)
    
class ActuallyDeepNetwork(nn.Module):
    """ Experimenting with a model with an (arbitrary) large capacity (it will probably overfit)"""
    def __init__(self, n_in, n_out, dropout=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_in, 8192),
            nn.LayerNorm(8192),
            nn.ReLU(),

            nn.Dropout(p=dropout),
            nn.Linear(8192, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(),

            nn.Dropout(p=dropout),
            nn.Linear(4096, n_out)
        )
 
    def forward(self, x):
        return self.model(x)