import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, backbone_channels, num_classes):
        super(MLPHead, self).__init__()
        in_channels = backbone_channels[-1]
        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_pred(x)
 