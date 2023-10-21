#./Abhisek/Super_Resolution/MFSR/model/MFSR.py

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.extend(['./'])

from FeatureExtraction import Decoder_Encoder
from UnitBlock import Unit
from . import Common


#############-----Model::Multi-Feature_Super_Resolution(MFSR)-----#############
class Model(nn.Module):
    def __init__(self, resolutions, scale_256, scale_64, resolution_in, num_unit_BC, num_unit_AC, channel_in_DE, channel_base_DE):  
        super(Model, self).__init__()

        '''
        ''Full Model For Multi-feature Super-resolution''
            [INPUTS]::
            resolutions:     [2, 4, 8]  Resolution Value list for final output                 
            scale_256:       [4, 2, 1]  Feature scaling factor list for downsample            
            scale_64:        [1, 2, 4]  Feature csaling factor list for upsample   
            resolution_in    4  Input resolution from user side from resolutions list             
            chennel_list:    [16, 32, 64, 128] Output channel list fron Decoder-Encoder module  
            num_unit_BC:     1  Number of units used before concat
            num_unit_AC:     2  Number of units used after concat
            channel_int:     Intermediate channel list for parameter reduction if want to
            channel_in_DE:   3  Input cannels for Decoder-Encoder module
            channel_base_DE: 16 Base-channels for Decoder-Encoder module

        '''

        ##self.arguments
        self.resolutions  = resolutions           
        self.scale_256 = scale_256                
        self.scale_64 = scale_64                 
        self.resolution_in = resolution_in
        self.num_unit_BC = num_unit_BC
        self.num_unit_AC = num_unit_AC
        channel_int = 16
        self.channel_in_DE = channel_in_DE
        self.channel_base_DE = channel_base_DE
        # chennel_list = [channel_base_DE, 2*channel_base_DE, 4*channel_base_DE, 8*channel_base_DE]   

        ##self.class
        self.Decoder_Encoder = Decoder_Encoder(channel_in = self.channel_in_DE, channel_base = self.channel_base_DE)
        ####----8 x (units)----####
        self.Unit1 = Unit(channel_base_DE, channel_int)
        self.Unit2 = Unit(2*channel_base_DE, channel_int)
        self.Unit3 = Unit(4*channel_base_DE, channel_int)
        self.Unit4 = Unit(8*channel_base_DE, channel_int)
        self.Unit_down = Unit(2*channel_base_DE, channel_int)
        self.Unit_up = Unit(8*channel_base_DE, channel_int)
        self.ConvBlock = Common.conv_block(8*channel_base_DE, channel_in_DE, kernel_size=1, act_type='lrelu')
        self.ConvLayer = Common.conv_layer(channel_in_DE, channel_in_DE, kernel_size=3)
        

    def Interpolate(self, x_int, scale_factor, mode='bicubic'):
        x_int = F.interpolate(x_int, scale_factor=scale_factor, mode=mode)
        return x_int
    
    def Upsampler(self, x_up, in_features, out_features, scale):
        if(scale == 1 or scale == 2 or scale == 4):
          x_up = nn.Conv2d(in_features, (scale*scale)*in_features, 1, 1, 0)(x_up)
          x_up = nn.PixelShuffle(scale)(x_up)
          x_up = nn.Conv2d(in_features, out_features, 1, 1, 0)(x_up)
          return x_up
        elif(scale == 1.5):
          # upsample by 3, then downsample by 2
          x_up = nn.Conv2d(in_features, (9)*in_features, 1, 1, 0)(x_up)
          x_up = nn.PixelShuffle(3)(x_up)
          x_up = self.Interpolate(x_up, 1/2),
          x_up = nn.Conv2d(in_features, out_features, 1, 1, 0)(x_up)
          return x_up
    
    def Downsampler(self, x_down, in_features, out_features, scale):
        if(scale == 1 or scale == 2 or scale == 4):
          x_down = self.Interpolate(x_down, 1/scale)
          x_down = nn.Conv2d(in_features, out_features,1 ,1 ,0)(x_down)
          return x_down
        elif(scale == 1.5):
          x_down = nn.Conv2d(in_features, 4*in_features, 1, 1, 0)(x_down)
          x_down = nn.PixelShuffle(2)(x_down)
          x_down = self.Interpolate(x_down, 1/3)
          x_down = nn.Conv2d(in_features, out_features, 1, 1, 0)(x_down)
          return x_down

    def OUT(self, out_down, out_up):
        out = torch.cat((out_down, out_up), dim = 1)
        out = self.ConvBlock(out)
        out = self.ConvLayer(out)
        output = self.upsampler(out)
        # self.GAP
        # self.ReLU
        # self.BN
        # self.dropout
        return output

    def forward(self, x):                               #[(32, 3, 64, 64)]
        x_features = self.Decoder_Encoder(x)     #[(3, 16, 64, 64), (3, 32, 128, 128), (3, 64, 256, 256), (3, 128, 512, 512)]

        ##16
        x_freq1, x_spat1 = self.unit1(x_features[0])
        c1, c2 = x_freq1.shape[1], x_spat1.shape[1] 
        x_freq1 = self.Upsampler(x_freq1, c1, 2*c1, 2)
        x_spat1 = self.Upsampler(x_spat1, c2, 2*c2, 2)
        ##32
        x_freq2, x_spat2 = self.unit2(x_features[1])
        ##64
        x_freq3, x_spat3 = self.unit3(x_features[2])
        c1, c2 = x_freq3.shape[1], x_spat3.shape[1] 
        x_freq3 = self.Upsampler(x_freq3, c1, 2*c1, 2)
        x_spat3 = self.Upsampler(x_spat3, c2, 2*c2, 2)
        ##128
        x_freq4, x_spat4 = self.unit4(x_features[3])

        out_freq_12, out_spat_12 = (x_freq1+x_freq2), (x_spat1+x_spat2)
        out_freq_34, out_spat_34 = (x_freq3+x_freq4), (x_spat3+x_spat4)

        out_down_freq, out_down_spat = self.unit_down(out_freq_12, out_spat_12)
        out_up_freq, out_up_spat = self.unit_up(out_freq_34, out_spat_34)

        out_down = torch.cat([out_down_freq, out_down_spat], dim = 1)
        out_up = torch.cat([out_up_freq, out_up_spat], dim = 1)

        out = self.OUT(out_down, out_up)    #self-attention module
        return out


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])



if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    arg = {'resolutions': [2, 4, 8], 'scale_256': [4, 2, 1], 
           'scale_64': [1, 2, 4], 'resolution_in': 4, 'num_unit_BC': 1, 
           'num_unit_AC': 2, 'channel_in_DE': 3, 'channel_base_DE':8}
    t = torch.rand(3, 3, 64, 64)
    model = Model(**arg)
    print(count_parameters(model))
    x = model(t)
    print(len(x))