import torch.nn as nn

import cnn3

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = cnn3.CNNArchitecture()

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
        )

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        return x