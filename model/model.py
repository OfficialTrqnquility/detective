
import torch


class Model(torch.nn.Module):
    def __init__(self, configuration):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(configuration.input_layers, 1, configuration.hidden_layers)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        print(x)
        # Define the forward pass of the model
        x, _ = self.lstm(x)
        logits = self.sigmoid(x)
        return logits
