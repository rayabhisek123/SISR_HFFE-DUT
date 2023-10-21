#./Abhisek/Super_Resolution/MFSR/model/MFSR.py

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.extend(['./'])



#############-----Decoder_Encoder_Module-----#############
class Decoder_Encoder(nn.Module):
  def __init__(self, channel_in, channel_base):  #channel_in=3, channel_base=16
    super(Decoder_Encoder, self).__init__()

    self.decoder_layer = [nn.Conv2d(channel_in,channel_base,1),
                          nn.ConvTranspose2d(channel_base, 2*channel_base,2 , 2),
                          nn.ConvTranspose2d(2*channel_base, 4*channel_base,2 , 2),
                          nn.ConvTranspose2d(4*channel_base, 8*channel_base,2 , 2)]
    self.dec = nn.Sequential(*self.decoder_layer)
    self.encoder_layer = [nn.Conv2d(8*channel_base,8*channel_base,1),
                          nn.Conv2d(8*channel_base, 4*channel_base, 2, 2),
                          nn.Conv2d(4*channel_base, 2*channel_base, 2, 2),
                          nn.Conv2d(2*channel_base, channel_base, 2, 2)]
    self.enc = nn.Sequential(*self.encoder_layer)

  def stack_and_squeeze(self, x1, x2):
    return torch.mean(torch.stack((x1, x2), 1), 1)

  def forward(self, x):              # x (3, 16, 32, 32)
    # decoder
    decoder = []                  
    for layer in self.decoder_layer:
      x = layer(x)
      decoder.append(x)              #[([N, 16, 32, 32]), ([N, 32, 64, 64]), ([N, 64, 128, 128]), ([N, 128, 256, 256])]
    
    # encoder
    encoder = []
    for layer in self.encoder_layer:
      x = layer(x)
      encoder.append(x)              #[([N, 128, 256, 256]), ([N, 64, 128, 128]), ([N, 32, 64, 64]), ([N, 16, 32, 32])]
    
    #adding similar outputs
    out = []
    encoder.reverse()
    for dec, enc in zip(decoder, encoder):
      x = self.stack_and_squeeze(dec, enc)
      out.append(x)
    for o in out:
      print(o.shape)
    return out









#############-----Frequency_Feature_Extraction_Module-----#############
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FFEM(nn.Module):
    def __init__(self, channel_in):
        super(FFEM, self).__init__()

        self.encoder = one_module(channel_in)
        self.decoder_low = one_module(channel_in) 
        #self.decoder_low = nn.Sequential(one_module(n_feats), one_module(n_feats), one_module(n_feats))
        self.decoder_high = one_module(channel_in)
        self.alise = one_module(channel_in)
        self.alise2 = BasicConv(2*channel_in, channel_in, 1,1,0) #one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(channel_in)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size = x.size()[-2:], mode='bilinear', align_corners=True)
        for i in range(1):
            x2 = self.decoder_low(x2)
        x3 = x2
        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size = x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4,high1],dim=1))))+ x


class one_module(nn.Module):
    def __init__(self, n_feats):
        super(one_module, self).__init__()
        self.layer1 = one_conv(n_feats, n_feats//2,3)
        self.layer2 = one_conv(n_feats, n_feats//2,3)
        # self.layer3 = one_conv(n_feats, n_feats//2,3)
        self.layer4 = BasicConv(n_feats, n_feats, 3,1,1)
        self.alise = BasicConv(2*n_feats, n_feats, 1,1,0)
        self.atten = CALayer(n_feats)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)
        self.weight3 = Scale(1)
        self.weight4 = Scale(1)
        self.weight5 = Scale(1)
    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # pdb.set_trace()
        x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2),self.weight3(x1)],1))))
        return self.weight4(x)+self.weight5(x4)


class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3, relu = True):
        super(one_conv,self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate,inchanels,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        if relu:
            self.relu = nn.PReLU(growth_rate)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)
    def forward(self,x):
        if self.flag == False:
            output = self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
        else:
            output = self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))
        return output#torch.cat((x,output),1)
        
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0,fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                                     nn.Sigmoid())
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    

#############-----Frequency_Feature_Extraction_Module-----#############
class SFEM(nn.Module):
  def __init__(self, channel_in, channel_int):
    super(SFEM, self).__init__()
    self.channel_in = channel_in
    channel_out = channel_in
    self.channel_int = channel_int
    # num_features_m = 32
    
    self.conv11_in_out = nn.Conv2d(channel_in, channel_out, 1, 1, 0)
    self.conv33_in_out = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
    self.conv55_in_out = nn.Conv2d(channel_in, channel_out, 5, 1, 2)
    self.conv11_in_int = nn.Conv2d(channel_in, channel_int, 1, 1, 0)
    self.conv33_int_int = nn.Conv2d(channel_int, channel_int, 3, 1, 1)
    self.conv55_int_out = nn.Conv2d(channel_int, channel_out, 5, 1, 2)
    self.conv13_in_int = nn.Conv2d(channel_in, channel_int, (1, 3), 1, (0, 1))
    self.conv31_int_out = nn.Conv2d(channel_int, channel_out, (3, 1), 1, (1, 0))
    self.conv15_in_int = nn.Conv2d(channel_in, channel_int, (1, 5), 1, (0, 2))
    self.conv51_int_out = nn.Conv2d(channel_int, channel_out, (5, 1), 1, (2, 0))
    self.BN = nn.BatchNorm2d(channel_out, eps=1e-5, momentum=0.01, affine=True)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()

  ############-----up_parallel-----############
  def up_parallel(self, x):
    x1 = self.conv11_in_out(x)
    x2 = self.conv33_in_out(x)
    x3 = self.conv55_in_out(x)
    x = (x1 + x2 + x3)
    return x

  ############-----up_series-----############
  def up_series(self, x):
    x = self.conv11_in_int(x)
    x = self.conv33_int_int(x)
    x = self.conv55_int_out(x)
    return x

  ############-----down_add-----############
  def down_add(self, x):
    x1 = self.conv13_in_int(x)
    x1 = self.conv31_int_out(x1)
    x2 = self.conv15_in_int(x)
    x2 = self.conv51_int_out(x2)
    return x

  ############-----down_cat-----############
  def down_cat(self, x):
    x1 = self.conv13_in_int(x)
    x2 = self.conv15_in_int(x)
    x = torch.cat((x1,x2),0)
    x = torch.mean(x, 0, True)
    x = torch.squeeze(x, 0)
    x1 = self.conv31_int_out(x)
    x2 = self.conv51_int_out(x)
    x = torch.cat((x1,x2),0)
    x = torch.mean(x, 0, True)
    x = torch.squeeze(x, 0)
    return x
  
  
  ##################-----Forward-----##################
  def forward(self, x):
    x_res = x
    x_up = torch.matmul(self.relu(self.up_parallel(x)), self.sigmoid(self.up_series(x)))
    x_up = self.BN(x_up)
    x_down = torch.matmul(self.relu(self.down_add(x)), self.sigmoid(self.down_cat(x)))
    x_down = self.BN(x_down)
    x = self.relu(x_up + x_down + x_res)
    return x
  





def count_parameters(model):
  """Count the number of parameters in a model."""
  return sum([p.numel() for p in model.parameters()])


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    t = torch.rand(3, 3, 8, 8)
    model = Decoder_Encoder(3, 8)
    print(count_parameters(model))
    x = model(t)
    print(len(x))