import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, cnn : nn.Module):
        super().__init__()
        self.cnn = cnn()

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
        )

        self.final_layer = nn.Linear(128, 1)
        
    def forward(self, x):
        B, T, C, H, W, D = x.shape

        x = x.view(B * T, C, H, W, D) 
        x = self.cnn(x)

        x = x.view(B, T, 128)
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.final_layer(x)

        return x

