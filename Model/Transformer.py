#./Abhisek/Super_Resolution/MFSR/model/MFSR.py

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.extend(['./'])

from TransUtils import Distributive, UnDistributive

###---Transformer_with_Unfold+Fold---###
class Trans(nn.Module):            
    def __init__(self, channel_in, dim=1156, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                  drop=0., attn_drop=0., drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(Trans, self).__init__()
        self.dim = dim
        
        self.DMHA = DMHA(self.dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        # self.NORM1 = nn.LayerNorm(self.dim)
        self.MLP = MLP(in_features=dim, hidden_features=dim//4, act_layer=act_layer, drop=drop)
        # self.NORM2 = nn.LayerNorm(self.dim)

    def Norm(self, x):
        N, HW, CKK = x.shape
        x_norm = nn.LayerNorm([HW, CKK])(x)   #H=Height, W=Width, C=Channel, K=Kernels of the patch0
        return x_norm
        
    def forward(self, x1, x2):
        b,c,h,w = x.shape     

        ##--Unfolding for Extract Patches--##                                                          #x=[128, 32, 34, 34] $$ B=128
        x = extract_image_patches(x, k_size=[3, 3], stride_list=[1,1], rates=[1, 1], padding='same')   #[128, 288, 1156]   ###Unfolding
        x = x.permute(0,2,1)                                                                       #[128, 1156, 288]

        ##--Tranformer Operation--##
        x = x + self.DMHA(self.NORM(x))                                                           ###Normalization + Multi-Head Attention
        x = x + self.MLP(self.NORM(x)) #self.drop_path(self.mlp(self.norm2(x)))                   ###Normalization + Multi-layer Perceptron

        ##--folding for get back image from patches--##
        x = x.permute(0,2,1)                                                                       #[128, 288, 1156]
        x = reverse_patches(x, (h,w), (3,3), 1, 1)                                                 #[128, 32, 34, 34]
        
        return x1, x2


def same_padding(x_patch, k_size, stride_list, rates):    #For extract_image_patches(below-1)
    assert len(x_patch.size()) == 4
    batch_size, channel, rows, cols = x_patch.size()
    out_rows = (rows + stride_list[0] - 1) // stride_list[0]
    out_cols = (cols + stride_list[1] - 1) // stride_list[1]
    effective_k_row = (k_size[0] - 1) * rates[0] + 1
    effective_k_col = (k_size[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*stride_list[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*stride_list[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    x_patch = torch.nn.ZeroPad2d(paddings)(x_patch)
    return x_patch

def extract_image_patches(x_patch, k_size, stride_list, rates, padding='same'):  #unfolding operation before entry to transformer
    assert len(x_patch.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = x_patch.size()
    
    if padding == 'same':
        x_patch = same_padding(x_patch, k_size, stride_list, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError(f'{padding} padding type is not supported. Enter "same" or "valid" ')

    unfold = torch.nn.Unfold(kernel_size=k_size, dilation=rates, padding=0, stride=stride_list)
    x_patch = unfold(x_patch)
    return x_patch  # [N, C*k*k, L], L is the total number of such blocks

def reverse_patches(x_patch, out_size, k_size, stride_list, padding):            #folding operation on data fetched from transformer
    unfold = torch.nn.Fold(output_size = out_size, kernel_size=k_size, dilation=1, padding=padding, stride=stride_list)
    x_patch = unfold(x_patch)
    return x_patch  # [N, C*k*k, L], L is the total number of such blocks


class DMHA(nn.Module):                                                                  ###Multi-head Attention
    def __init__(self, dim, num_heads=8, qkv_bias=False, drop_out_DMHA=False):     #dim=288
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5    #NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        
        self.reduce = nn.Linear(dim, dim//2, bias=qkv_bias)
        self.qkv = nn.Linear(dim//2, dim//2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim//2, dim)
        self.DS = Distributive(N=289, num_heads=8)
        self.UDS = UnDistributive(N=289, num_heads=8)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        if drop_out_DMHA:
            self.attn_drop = nn.Dropout(0.5)
        else:
            self.attn_drop = nn.Dropout(0.)


    def forward(self, x1, x2):            #[128, 1156, 288]
        ####---For_x1---####
        x1 = self.reduce(x1)           #[128, 1156, 144]
        B, N, C = x1.shape
        qkv = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  #[128, 1156, 432]-->[128, 1156, 432]-->[128, 1156, 3, 8, 18]-->[3, 128, 8, 1156, 18]
        qkv=[torch.chunk(a, 4, dim=-2) for a in qkv]         #[3, 128, 8, 1156, 18]-->List[Tuple(q1, q2, q3, q4), Tuple(k1, k2, k3, k4), Tuple(v1, v2, v3, v4)] #All r size of [128, 8, 289, 18]                                                         
        q, k, v = qkv[0], qkv[1], qkv[2]        #q_all=Tuple(Tensor([128, 8, 289, 18]), Tensor([128, 8, 289, 18]), Tensor([128, 8, 289, 18]), Tensor([128, 8, 289, 18]))  
        output = []
        for q,k,v in zip(q, k, v):
            attn = self.DS(q, k)
            attn = attn.softmax(dim=-1)                               #[128, 8, 289, 289]-->[128, 8, 289, 289]
            attn = self.attn_drop(attn)                               #[128, 8, 289, 289]-->[128, 8, 289, 289]
            trans_x1 = (attn @ v).transpose(1, 2)#.reshape(B, N, C)   #[128, 8, 289, 289]@[128, 8, 289, 18]=[128, 8, 289, 18]-->[128, 289, 8, 18]
            output.append(trans_x1) 
        x1 = torch.cat(output,dim=1)                                  #[128, 289, 8, 18]-->[128, 1156, 8, 18]
        x1 = x1.reshape(B,N,C)                                        #[128, 1156, 8, 18]-->[128, 1156, 144]
        x1 = self.proj(x1)                                            #[128, 1156, 144]-->#[128, 1156, 288]

        ####---For_x2---####
        x2 = self.reduce(x2)           #[128, 1156, 144]
        B, N, C = x2.shape
        qkv = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #[128, 1156, 432]-->[128, 1156, 432]-->[128, 1156, 3, 8, 18]-->[3, 128, 8, 1156, 18]
        qkv=[torch.chunk(a, self.chunk_size, dim=-2) for a in qkv]  #[3, 128, 8, 1156, 18]-->List[Tuple(q1, q2, q3, q4), Tuple(k1, k2, k3, k4), Tuple(v1, v2, v3, v4)] #All r size of [128, 8, 289, 18]
        q, k, v = qkv[0], qkv[1], qkv[2]        #q_all=Tuple(Tensor([128, 8, 289, 18]), Tensor([128, 8, 289, 18]), Tensor([128, 8, 289, 18]), Tensor([128, 8, 289, 18]))  
        output = []
        for q,k,v in zip(q, k, v):
            q = self.UDS(q)
            k = self.UDS(k)
            k = torch.einsum('ijkl->ijlk', k)                         #[128, 8, 289, 18]-->[128, 8, 18, 289]
            attn = torch.einsum('mnop,ijpk->mnok', q, k)* self.scale  #[128, 8, 289, 18] @ [128, 8, 18, 289] = [128, 8, 289, 289] 
            #attn = (q @ k.transpose(-2, -1)) * self.scale            #[128, 8, 289, 18] @ [128, 8, 18, 289] = [128, 8, 289, 289] 
            attn = attn.softmax(dim=-1)                               #[128, 8, 289, 289]-->[128, 8, 289, 289]
            attn = self.attn_drop(attn)                               #[128, 8, 289, 289]-->[128, 8, 289, 289]
            trans_x2 = (attn @ v).transpose(1, 2)#.reshape(B, N, C)   #[128, 8, 289, 289]@[128, 8, 289, 18]=[128, 8, 289, 18]-->[128, 289, 8, 18]
            output.append(trans_x2)   
        x2 = torch.cat(output,dim=1)                                  #[128, 289, 8, 18]-->[128, 1156, 8, 18]
        x2 = x2.reshape(B,N,C)                                        #[128, 1156, 8, 18]-->[128, 1156, 144]
        x2 = self.proj(x2)                                            #[128, 1156, 144]-->#[128, 1156, 288]
        return x1, x2      #x1 = Distributive, x2 = Undistributive
    

class MLP(nn.Module):                  ###Can be changed, so look at this direction
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features//4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

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
    model = DMHA(channel_in, channel_int)
    print(count_parameters(model))
    x = model(t)
    print(len(x))