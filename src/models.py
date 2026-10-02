import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=384, num_classes=5, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x_pooled = torch.mean(x, dim=1)
        return self.net(x_pooled)

class TemporalGRU(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=5, num_layers=1, dropout=0.4):
        super().__init__()
        self.input_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.input_drop(x)
        out, _ = self.gru(x)
        last_step_out = out[:, -1, :]
        return self.classifier(last_step_out)
