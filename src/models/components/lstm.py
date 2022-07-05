from torch import nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        latent_dim: int = 256,
        num_layers: int = 4,
        bidirectional=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            dropout=0.1,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.fc_in_features = latent_dim * 2 if bidirectional else latent_dim
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.fc_in_features, out_features=256),
            nn.BatchNorm1d(101),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(101),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        return self.fc(lstm_out)
