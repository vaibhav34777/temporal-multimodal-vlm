import math
import torch
import torch.nn as nn


class SingleFrameBaseline(nn.Module):
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
        center_frame = x[:, x.shape[1] // 2, :]
        return self.net(center_frame)


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


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=384, num_heads=4, num_layers=2, ff_dim=512,
                 num_classes=5, dropout=0.4, use_positional_encoding=True, max_seq_len=16):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.input_proj = nn.Linear(input_dim, input_dim)

        if use_positional_encoding:
            pe = torch.zeros(max_seq_len, input_dim)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-math.log(10000.0) / input_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        if self.use_positional_encoding:
            x = x + self.pe[:, :x.size(1), :]
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)
