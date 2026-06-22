import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock2D(nn.Module):

    """
    small Resnet-style 2D residual block
    Input: [N, Cin, H, W]
    Output: [N, Cout, H/stride, W/stride]
    """

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, bias=False)
        
        

