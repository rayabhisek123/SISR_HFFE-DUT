#./Abhisek/Super_Resolution/MFSR/model/MFSR.py

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.extend(['./'])

from . import Transformer

#import Transformer

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
    def __init__(self, channel_in, DownFactor=2, UpFactor=2):
        super(CAM, self).__init__()

        #Arguments
        self.channel_in  = out_channel = channel_in
        self.out_channel = out_channel
        self.DownFactor = DownFactor
        self.UpFactor = UpFactor


        #For_Forward
        self.ConvDown1 = nn.Conv2d(self.channel_in, self.channel_in//DownFactor, kernel_size=3, padding=1)
        self.ConvDown2 = nn.Conv2d(self.channel_in, self.channel_in//DownFactor, kernel_size=3, padding=1)
        self.ConvDown3 = nn.Conv2d(self.channel_in, self.channel_in, kernel_size=3, stride=2, padding=1)
        self.ConvDown4 = nn.Conv2d(self.channel_in, self.channel_in, kernel_size=3, stride=2, padding=1)
        self.Up_Feature1 = nn.ConvTranspose2d(self.channel_in, self.channel_in, kernel_size=3, stride=2, padding=1)
        self.Up_Feature2 = nn.ConvTranspose2d(self.channel_in, self.channel_in, kernel_size=3, stride=2, padding=1)
        self.Up_Channel1 = nn.Conv2d(self.channel_in//DownFactor, self.channel_in, kernel_size=3, padding=1)
        self.Up_Channel2 = nn.Conv2d(self.channel_in//DownFactor, self.channel_in, kernel_size=3, padding=1)
        self.BN1 = self.BN = nn.BatchNorm2d(self.channel_in)
        self.BN2 = self.BN = nn.BatchNorm2d(self.channel_in)
        self.BN3 = self.BN = nn.BatchNorm2d(self.channel_in)
        self.BN4 = self.BN = nn.BatchNorm2d(self.channel_in)
        self.BN5 = self.BN = nn.BatchNorm2d(self.channel_in)
        self.BN6 = self.BN = nn.BatchNorm2d(self.channel_in)
        self.BN7 = self.BN = nn.BatchNorm2d(self.channel_in)
        self.BN8 = self.BN = nn.BatchNorm2d(self.channel_in)
        self.relu = nn.ReLU()

    ##########################
    def forward(self, x_1, x_2):
        '''
        x_1 : output from transformer 1
        x_2 : output from transformer 2
        '''
        B1, C1, H1, W1 = x_1.size()
        B2, C2, H2, W2 = x_2.size()
        
        ##[C x C]
        x_1c_1 = x_1.view(B1, C1, -1)
        x_1c_2 = x_1.view(B1, C1, -1).permute(0, 2, 1).contiguous()
        x_1c = torch.bmm(x_1c_1, x_1c_2)  #[3, 16, 16]
        
        x_2c_1 = x_2.view(B2, C2, -1)
        x_2c_2 = x_2.view(B2, C2, -1).permute(0, 2, 1).contiguous()
        x_2c = torch.bmm(x_2c_1, x_2c_2)  #[3, 16, 16]

        ##[HW x HW]                           
        x_1hw_1 = x_1.view(B1, C1, -1).permute(0, 2, 1).contiguous()
        x_1hw_2 = x_1.view(B1, C1, -1)
        x_1hw = torch.bmm(x_1hw_1, x_1hw_2)   #[3, 1024, 1024]    
                               
        x_2hw_1 = x_2.view(B2, C2, -1).permute(0, 2, 1).contiguous()
        x_2hw_2 = x_2.view(B2, C2, -1)
        x_2hw = torch.bmm(x_2hw_1, x_2hw_2)   #[3, 1024, 1024]

        ##[C/2 x C/2]  #cds=channel downsample
        x_1_cds = self.ConvDown1(x_1)                   #[3, 16, 32, 32]-->[3, 8, 32, 32] 
        x_1_cds1 = x_1_cds.view(B1, C1//self.DownFactor, -1)
        x_1_cds2 = x_1_cds.view(B1, C1//self.DownFactor, -1).permute(0, 2, 1).contiguous()
        x_1cds = torch.bmm(x_1_cds1, x_1_cds2)  #[3, 8, 8]
        
        x_2_cds = self.ConvDown2(x_2)                   #[3, 16, 32, 32]-->[3, 8, 32, 32] 
        x_2_cds1 = x_2_cds.view(B2, C2//self.DownFactor, -1)
        x_2_cds2 = x_2_cds.view(B2, C2//self.DownFactor, -1).permute(0, 2, 1).contiguous()
        x_2cds = torch.bmm(x_2_cds1, x_2_cds2)  #[3, 8, 8]
        

        #[HW/4 x HW/4]
        x_1_hwds = self.ConvDown3(x_1)                   #[3, 16, 32, 32]-->[3, 16, 16, 16]      
        x_1_hwds1 = x_1_hwds.view(B1, C1, -1).permute(0, 2, 1).contiguous()
        x_1_hwds2 = x_1_hwds.view(B1, C1, -1)
        x_1hwds = torch.bmm(x_1_hwds1, x_1_hwds2)        #[3, 256, 256]
        
        x_2_hwds = self.ConvDown4(x_2)                   #[3, 16, 16, 16]      
        x_2_hwds1 = x_2_hwds.view(B2, C2, -1).permute(0, 2, 1).contiguous()
        x_2_hwds2 = x_2_hwds.view(B2, C2, -1)
        x_2hwds = torch.bmm(x_2_hwds1, x_2_hwds2)        #[3, 256, 256]
        
        
        #Feature processsing
        self_HW = self.relu(x_1 + self.BN1(torch.matmul(x_1.view(B1, C1, -1), x_1hw).view(B1, C1, H1, W1)))
        cross_HW = self.relu(x_1 + self.BN2(torch.matmul(x_1.view(B1, C1, -1), x_2hw).view(B1, C1, H1, W1)))
        down_self_HW = self.relu(x_1 + self.BN3(self.Up_Feature1(torch.matmul(x_1_hwds.view(B1, C1, -1), x_1hwds).view(B1, C1, H1//self.DownFactor, W1//self.DownFactor), output_size=x_1.size())))
        down_cross_HW = self.relu(x_1 + self.BN4(self.Up_Feature2(torch.matmul(x_1_hwds.view(B1, C1, -1), x_2hwds).view(B1, C1, H1//self.DownFactor, W1//self.DownFactor), output_size=x_1.size())))
        self_feature1 = self_HW + down_self_HW
        cross_feature1 = cross_HW + down_cross_HW
        out_HW = torch.mean(torch.stack((self_feature1, cross_feature1), 1), 1)
        
        
        #Channel processing
        self_C = self.relu(x_1 + self.BN5(torch.matmul(x_1c, x_1.view(B1, C1, -1)).view(B1, C1, H1, W1)))
        cross_C = self.relu(x_1 + self.BN6(torch.matmul(x_2c, x_1.view(B1, C1, -1)).view(B1, C1, H1, W1)))
        down_self_C = self.relu(x_1 + self.BN7(self.Up_Channel1(torch.matmul(x_1cds, x_1_cds.view(B1, C1//self.DownFactor, -1)).view(B1, -1, H1, W1))))
        down_cross_C = self.relu(x_1 + self.BN8(self.Up_Channel2(torch.matmul(x_2cds, x_1_cds.view(B1, C1//self.DownFactor, -1)).view(B1, -1, H1, W1))))
        self_feature2 = self_C + down_self_C
        cross_feature2 = cross_C + down_cross_C
        out_C = torch.mean(torch.stack((self_feature2, cross_feature2), 1), 1)
        #Combined output
        out = torch.mean(torch.stack((out_HW, out_C), 1), 1)
        return out





def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    t1 = torch.rand(3, 16, 32, 32)
    t2 = torch.rand(3, 16, 32, 32)
    model = CAM(16, DownFactor=2, UpFactor=2)
    print(count_parameters(model))
    x1 = model(t1, t2)
    print(x1.shape)