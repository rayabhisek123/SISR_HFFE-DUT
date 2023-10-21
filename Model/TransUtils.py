#./Abhisek/Super_Resolution/MFSR/model/MFSR.py

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.extend(['./'])


class Distributive(nn.Module):
    def __init__(self, dim=1156, num_heads=8, N=289, C=18, multiplicative_mask = True):
        super().__init__()
        #self.M_add = M_add
        #self.M_mul = M_mul
        self.N = N
        self.C = C
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.Conv1 = nn.Conv1d(self.C, self.C,  1, stride=1, padding="same")
        nn.init.xavier_normal_(self.Conv1.weight)
        nn.init.constant_(self.Conv1.bias, 0)
        
        self.Conv2 = nn.Conv1d(self.C, self.C,  3, stride=1, padding="same")
        nn.init.xavier_normal_(self.Conv2.weight)
        nn.init.constant_(self.Conv2.bias, 0)
        
        self.Conv3 = nn.Conv1d(self.C, self.C,  5, stride=1, padding="same")
        nn.init.xavier_normal_(self.Conv3.weight)
        nn.init.constant_(self.Conv3.bias, 0)
        
        
        #self.relu = nn.ReLU(inplace=True)
        self.BN = self.bn = nn.BatchNorm2d(self.num_heads, eps=1e-5, momentum=0.01, affine=True)
        ####%%%%--Mask_Multiplicative--%%%%####
        # if multiplicative_mask:
        #     self.m_mask = lambda x: x * nn.Parameter(torch.ones(x.size()).to(x.get_device()))
        # else :
        #     self.m_mask = lambda x: x
        
    def forward(self, q, k):                                                       #[128, 8, 289, 18]
        assert q.size() == k.size()
        B, H, N, C = q.shape                                                    #B=Batch=128 H=Head=8 N=Data=289 C=Channel=18
        #print('size of q',q.size())
        #input()
        assert self.N == N
        assert self.C == C
        q = q.permute(0, 1, 3, 2).contiguous().view(B*H, C, N)                   #[128*8, 18, 289]
        k = k.permute(0, 1, 3, 2).contiguous().view(B*H, C, N)                   #[128*8, 18, 289]
        
        q1 = self.Conv1(q).permute(0, 2, 1).contiguous().view(-1, H, N, C)      #[128, 8, 289, 18]
        q2 = self.Conv2(q).permute(0, 2, 1).contiguous().view(-1, H, N, C)      #[128, 8, 289, 18]
        q3 = self.Conv3(q).permute(0, 2, 1).contiguous().view(-1, H, N, C)      #[128, 8, 289, 18]
        
        k1 = self.Conv1(k).view(-1, H, C, N)                                    #[128, 8, 18, 289]
        k2 = self.Conv2(k).view(-1, H, C, N)                                    #[128, 8, 18, 289]
        k3 = self.Conv3(k).view(-1, H, C, N)                                    #[128, 8, 18, 289]
        
        attn1 = torch.einsum('mnop,ijpk->mnok', q1, k1)* self.scale             #[128, 8, 289, 18]*[128, 8, 18, 289]-->[128, 8, 289, 289]
        attn1 = torch.unsqueeze(attn1, 0)                                       #[128, 8, 289, 289]-->[1, 128, 8, 289, 289]
        attn2 = torch.einsum('mnop,ijpk->mnok', q2, k2)* self.scale             #[128, 8, 289, 18]*[128, 8, 18, 289]-->[128, 8, 289, 289]
        attn2 = torch.unsqueeze(attn2, 0)                                       #[128, 8, 289, 289]-->[1, 128, 8, 289, 289]
        attn3 = torch.einsum('mnop,ijpk->mnok', q3, k3)* self.scale             #[128, 8, 289, 18]*[128, 8, 18, 289]-->[128, 8, 289, 289]
        attn3 = torch.unsqueeze(attn3, 0)                                       #[128, 8, 289, 289]-->[1, 128, 8, 289, 289]
        attn = torch.cat((attn1, attn2, attn3), 0)                              #[1, 128, 8, 289, 289]-->[3, 128, 8, 289, 289]
        attn = torch.mean(attn, 0, True)                                        #[3, 128, 8, 289, 289]-->[1, 128, 8, 289, 289]
        attn = torch.squeeze(attn, 0)                                           #[1, 128, 8, 289, 289]-->[128, 8, 289, 289]
        # attn = self.m_mask(attn)                                                #[128, 8, 289, 289]-->[128, 8, 289, 289]
        return self.BN(attn)                                                    #[128, 8, 289, 289]--8-->[128, 8, 289, 289
    


class UnDistributive(nn.Module):
    def __init__(self, N = 289, C=18, multiplicative_mask = True, additive_mask = True):  #channels_int = 32
        super().__init__()
        self.N = N
        self.C = C
        #self.C_int = channels_int        
        
        ####%%%%--Layers--%%%%####
        self.Conv1 = nn.Conv1d(self.C, self.C, 3, stride=1, padding="same")
        nn.init.xavier_normal_(self.Conv1.weight)
        nn.init.constant_(self.Conv1.bias, 0)
        
        self.Conv2 = nn.Conv1d(self.C, self.C, 5, stride=1, padding="same")
        nn.init.xavier_normal_(self.Conv2.weight)
        nn.init.constant_(self.Conv2.bias, 0)
        
        # self.Linear = nn.Linear(self.N, self.N, bias=True, device=None, dtype=None)
        # nn.init.kaiming_normal_(self.Linear.weight)
        # nn.init.constant_(self.Linear.bias, 0)
        
        self.relu = nn.ReLU(inplace=True)
        self.BN = self.bn = nn.BatchNorm1d(self.C, eps=1e-5, momentum=0.01, affine=True)
        
        # ####%%%%--Mask_Multiplicative--%%%%####
        # if multiplicative_mask:
        #     self.m_mask = lambda x: x * nn.Parameter(torch.ones(x.size()).to(x.get_device()))
        # else :
        #     self.m_mask = lambda x: x
        
        ####%%%%--Mask_Additive--%%%%####
        # if additive_mask:
        #     self.a_mask = lambda x: x + nn.Parameter(torch.zeros(x.size()).to(x.get_device()))  
        # else :
        #     self.a_mask = lambda x: x
        
    def forward(self, x):                                                       #[128, 8, 289, 18]
        B, H, N, C = x.shape                                                    #B=Batch=128 H=Head=8 N=Data=289 C=Channel=18
        assert self.N == N
        assert self.C == C
        x = x.permute(0, 1, 3, 2).contiguous()                                  #[128, 8, 18, 289]
        x_in = x.view(B*H, C, N)                                                #[128*8, 18, 289]
        x13 = self.Conv1(x_in)                                                  #[128*8, 18, 289]
        x15 = self.Conv2(x_in)                                                  #[128*8, 18, 289]
        print(x.get_device())
        input()
        # x_p = self.a_mask(x13 + x15)#.view(B, H, C, N)                          #[128*8, 18, 289]  #.permute(0, 1, 3, 2).contiguous()
        x_p = x13 + x15
        x_p_out = self.relu(x_p)                                                #[128*8, 18, 289]
        
        x1315 = self.Conv2(self.Conv1(x_in))                                    #[128*8, 18, 289]
        x1315_permute = x1315.permute(0, 2, 1)                                  #[128*8, 289, 18]
        x1315 = x1315_permute @ x1315                                          #[128*8, 289, 289]
        x_s_out = x1315  
        #x_s_out = self.Linear(x1315)                                                #[128*8, 289, 289]
        # x_s = torch.mean(x_s, -2, True)                                         #[128*8, 1, 289]
        # x_s = torch.squeeze(x_s, -2)                                            #[128*8, 289]
        # x_s_out = x_s.view(B*H, -1, N)                                          #[128*8, 289, 289]
        # x_out = self.m_mask(x_p_out @ x_s_out)                                  #[128*8, 18, 289]*[128*8, 289, 289]-->[128*8, 18, 289]
        x_out = x_p_out @ x_s_out
        x_out = self.BN(x_out)                                                  #[128*8, 18, 289]
        x_out = x_out.view(-1, H, C, N).permute(0, 1, 3, 2).contiguous()        #[128, 8, 289, 18]
        return x_out
    



def count_parameters(model):
  """Count the number of parameters in a model."""
  return sum([p.numel() for p in model.parameters()])


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    t1 = torch.rand(128, 8, 289, 18)
    model = UnDistributive()
    print(count_parameters(model))
    x1 = model(t1)
    print(x1.shape)