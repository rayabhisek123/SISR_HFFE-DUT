#./Abhisek/Super_Resolution/MFSR/model/MFSR.py

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.extend(['./'])

from Attention import CAM
from Transformer import Trans
from FeatureExtraction import FFEM, SFEM

class Unit(nn.Module):
    def __init__(self, channel_in, channel_int):
        super(Unit, self).__init__()
        
        self.MFEM = MFEM(channel_in, channel_int, upscale = 1)       #MFEM --> Multi-modal Feature Extraction Module
        self.CAM = CAM(channel_in)                                             #CAM --> Cross Attention Module
        
    def forward(self, x_freq, x_spat):
        x_freq, x_freq = self.MFEM(x_freq, x_freq)
        x_freq, x_freq = self.CAM(x_freq, x_freq)
        return x_freq, x_freq
    
def default_conv(in_channels, out_channels, kernel_size, bias=True, groups = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, groups = groups)

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class MFEM(nn.Module):
    def __init__(self, channel_base_DE, channel_int):       #upscale=4
        super(MFEM, self).__init__()

        # channel_out = channel_in = x.shape[1]
        # model = SFEM(channel_in, channel_int)                                
        self.weight2 = Scale(1)
        self.weight1 = Scale(1)
        self.Trans = Trans(channel_in, dim=288) 
        self.FFEM = FFEM(channel_in)
        self.SFEM = SFEM(channel_in, channel_int)           #SFEM --> Spatial Fetaure Extraction Module

    def forward(self, x_freq, x_spat):
        out1, out2 = self.Trans(self.FFEM(x_freq), self.SFEM(x_spat))    #Feature_HPF, Feature_spatial = self.attention1(HPF, spatial)

        return self.weight1(out1), self.weight2(out2)
    



def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    channel_in = 3
    channel_int = 16
    t = torch.rand(3, 3, 64, 64)
    model = Unit(channel_in, channel_int)
    print(count_parameters(model))
    x = model(t)
    print(len(x))
    
