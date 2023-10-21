#./Abhisek/Super_Resolution/MFSR/model/MFSR.py

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.extend(['./'])

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, raw_feature_norm, smooth=9, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    queryT = torch.transpose(query, 1, 2)
    attn = torch.bmm(context, queryT)

    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    
    if weight is not None:
      attn = attn + weight

    attn_out = attn.clone()

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    attn = F.softmax(attn*smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext


#############-----Cross_Attention_Module-----#############
class CAM(nn.Module):
    '''
    Compute Cross-Attention Fetures between Frequency Stream and Spatial Stream

    '''
    def __init__(self, channel_in, DownFactor=2, UpFactor=4):
        super(CAM, self).__init__()

        #Arguments
        self.channel_in  = out_channel = channel_in
        self.out_channel = out_channel
        self.DownFactor = DownFactor
        self.UpFactor = UpFactor


        #For_inside_functions
        self.ConvDown = nn.Conv2d(self.channel_in, self.channel_in//DownFactor, 1, 1, 0)
        self.ConvUp = nn.Conv1d(self.channel_in//DownFactor, self.channel_in, 1, 1, 0)
        self.UpSample = nn.Upsample(scale_factor = UpFactor, mode='bicubic')
        # self.UpFeature = nn.Conv2d(channel_in, (DownFactor*DownFactor)*channel_in, 1, 1, 0)
        self.SoftMax = nn.Softmax(dim = -1)
        #For_forward
        # self.cam_channels = CAM_Calculate()
        # self.cam_hw = PAM_Module()
        # self.down_2_34 = Downsample(34, 2)
        # self.down_4_32 = Downsample(32, 4)
        # self.att = AttentionCalculation()
        # self.up_4_8 = UpSample(8, 4, 'noconv')
        # self.up_4_289 = UpSample(289, 4, 'noconv')
        self.bn = nn.BatchNorm2d(channel_in)
        self.relu = nn.ReLU(inplace = True)

        

    def Channel(self, x):
        """
        Input  : B X C X H X W
        Output : B X C X C
        """
        B, C, H, W = x.size()
        x1 = x.view(B, C, -1)
        x2 = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        x = torch.bmm(x1, x2)
        return x

    def Feature(self, x):
        """
        Input  :(B X C X H X W)
        Output : B X (HxW) X (HxW)
        """
        B, C, H, W = x.size()
        x1 = x.view(B, -1, W * H).permute(0, 2, 1).contiguous()
        x2 = x.view(B, -1, W * H).contiguous()
        x = torch.bmm(x1, x2)
        return x

    def Down_Channel(self, x):
        '''
        Input  : (B X C X H X W)
        Output : (B X C/4 X H X W)
        '''
        x = self.ConvDown(x)
        return x

    def Down_Feature(self, x):
        '''
        Input  : (B X C X Hr X Wr)
        Output : (B X C X Hr X Wr)
        '''
        B, C, H, W = x.shape
        x = F.interpolate(x, scale_factor=1/self.DownFactor, mode='bicubic')        
        return x
    
    def Up_Channel(self, x):
        '''
        Input  : (B X C/4 X C/4)
        Output : (B X C X C)
        '''
        x = self.ConvUp(self.ConvUp(x).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        return x

    def Up_Feature(self, x):
        '''
        Input  : (B X HW/4 X HW/4)
        Output : (B X HW X HW)
        '''
        x = torch.unsqueeze(x, 0)
        x = self.UpSample(x)
        x = torch.squeeze(x, 0)

        return x

    def Attention(self, x_1, x_2):
        '''
        Multiplicative Attention
        Input == Output
        '''
        out = self.SoftMax(torch.matmul(x_1, torch.transpose(x_2, 1, 2)))
        return out

    ##########################
    def forward(self, x_1, x_2):
        '''
        x_1 : output from transformer 1
        x_2 : output from transformer 2
        '''
        b, c, h, w = x_1.size()
        #[C x C]
        x_1_channel = self.Channel(x_1)                        #[3, 16, 16]
        x_2_channel = self.Channel(x_2)                        #[3, 16, 16]

        #[HW x HW]
        x_1_hw = self.Feature(x_1)                             #[3, 1024, 1024]
        x_2_hw = self.Feature(x_2)                             #[3, 1024, 1024]

        #[C/4 x C/4]
        x_1_channel_downsampled = self.Channel(self.Down_Channel(x_1))        #[3, 8, 8]
        x_2_channel_downsampled = self.Channel(self.Down_Channel(x_2))        #[3, 8, 8]

        #[HW/4 x HW/4]
        x_1_hw_downsampled = self.Down_Feature(x_1)              
        x_1_hw_downsampled = self.Feature(x_1_hw_downsampled)         #[3, 256, 256]
        x_2_hw_downsampled = self.Down_Feature(x_2)
        x_2_hw_downsampled = self.Feature(x_2_hw_downsampled)         #[3, 256, 256]

        channel_1_out = torch.bmm(self.Attention(x_1_channel, x_2_channel), self.Up_Channel(self.Attention(x_1_channel_downsampled, x_2_channel_downsampled)))  #[3, 16, 16], [3, 8, 8]
        channel_2_out = torch.bmm(self.Attention(x_2_channel, x_1_channel), self.Up_Channel(self.Attention(x_2_channel_downsampled, x_1_channel_downsampled)))
        hw_1_out = torch.bmm(self.Attention(x_1_hw, x_2_hw), self.Up_Feature(self.Attention(x_1_hw_downsampled, x_2_hw_downsampled)))   #[3, 1024, 1024], [3, 256, 256]
        hw_2_out = torch.bmm(self.Attention(x_2_hw, x_1_hw), self.Up_Feature(self.Attention(x_2_hw_downsampled, x_1_hw_downsampled)))
        channel_1_out = torch.bmm(channel_1_out, x_1.view(b, c, h*w)).view(b, c, h, w)
        channel_1_out = self.relu(self.bn(channel_1_out) + x_1)
        channel_2_out = torch.bmm(channel_2_out, x_2.view(b, c, h*w)).view(b, c, h, w)
        channel_2_out = self.relu(self.bn(channel_2_out) + x_2)
        hw_1_out = torch.bmm(hw_1_out, x_1.view(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).view(b, c, h, w)
        hw_1_out = self.relu(self.bn(hw_1_out) + x_1)
        hw_2_out = torch.bmm(hw_2_out, x_2.view(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).view(b, c, h, w)
        hw_2_out = self.relu(self.bn(hw_2_out) + x_2)
        res1 = torch.mean(torch.stack((channel_1_out, hw_1_out), 1), 1)
        res2 = torch.mean(torch.stack((channel_2_out, hw_2_out), 1), 1)
        print(res1.size(), res2.size())
        return res1, res2





def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    t1 = torch.rand(3, 16, 32, 32)
    t2 = torch.rand(3, 16, 32, 32)
    model = CAM(16, DownFactor=2, UpFactor=4)
    print(count_parameters(model))
    x1, x2 = model(t1, t2)
    print(x1.shape, x2.shape)