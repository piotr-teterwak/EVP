import torch


class MeanCenterLayer(torch.nn.Module):

    def __init__(self, num_features):
        super(MeanCenterLayer, self).__init__()
        self.register_buffer("running_mean", torch.zeros(num_features))

    def forward(self, input):
        out = input - self.running_mean
        if self.training:
            with torch.no_grad():
                self.running_mean = 0.9 * input + 0.1 * self.running_mean
        return out
