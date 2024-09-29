""" Unet full sturcture
    Full assembly of the parts to form the complete network 
Typical usage example:
    -
"""
import torch.nn.functional as F
from .unet_parts import *
from .tma_head import *
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.encoder = UNetEncoder(n_channels=n_channels,bilinear=bilinear)
        self.decoder = UNetDecoder(n_classes=n_classes,bilinear=bilinear)
    def forward(self,x,mem_list=None):
        x_list = self.encoder(x)
        logits = self.decoder(x_list)
        return logits 
    
    
class SeqUNet(nn.Module):
    def __init__(self, n_channels, n_classes,sequence_num, bilinear=True):
        super(SeqUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sequence_num = sequence_num
        factor = 2 if bilinear else 1
        self.encoder = UNetEncoder(n_channels=n_channels,bilinear=bilinear)
        self.decoder = UNetDecoder(n_classes,bilinear)
        self.head = TMAHead(sequence_num=self.sequence_num,
                            in_channels=1024//factor,
                            out_channels=1024//factor,
                            key_channels=256,
                            value_channels=1024)

    def forward(self,x, mem_list, y=None):
        """
        Parameters:
            x: video feature of size [B,C=3,H,W]
            y: force feature of size [B,T,L=6]
            mem_list: (list[Tensor]): len(sequence_imgs) is equal to sequence_num,
                each element is a Tensor with shape of BxCxHxW.
        return:
            logits: output mask of size [B,C=3,H,W]
        """
        # encode cur_frame
        x1,x2,x3,x4,x5 = self.encoder(x) # B C H W

        # encode mem_list
        x_mem = self.concat(mem_list) # B T C H W
        b,t,c,h,w = x_mem.shape
        x_mem = x_mem.reshape(b*t,c,h,w)
        _,_,_,_,x5_mem = self.encoder(x_mem)
        _,c_5,h_5,w_5 = x5_mem.shape
        x5_mem = x5_mem.reshape(b,t,c_5,h_5,w_5)
        x5_mem = torch.transpose(x5_mem,0,1) # T B C H W

        # fuse video features
        x_time = self.head(x5,x5_mem) #B C H W

        # decode
        logits = self.decoder([x1,x2,x3,x4,x_time])
        return logits
    
    def concat(self,mem_list):
        for i,_ in enumerate(mem_list):
            mem_list[i] = mem_list[i].unsqueeze(1)
        x_mem = torch.cat(mem_list,dim=1) 
        return x_mem


class FGUNet(nn.Module):
    """base TMAMMNet. fuse force after stage5. frc_ch=64 """
    def __init__(self, n_channels, n_classes,sequence_num, bilinear=True):
        super(FGUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.sequence_num = sequence_num
        self.encoder = UNetEncoder(n_channels=n_channels,bilinear=bilinear)
        self.decoder = UNetDecoder(n_classes,bilinear)
        self.head = TMAHeadForce(sequence_num=self.
                                 sequence_num,
                                 in_channels=1024//factor,
                                 out_channels=1024//factor,
                                 key_channels=256,
                                 value_channels=1024)

    def forward(self,x, mem_list, y):
        """
        Parameters:
            x: video feature of size [B,C=3,H,W]
            y: force feature of size [B,T,L=6]
            mem_list: (list[Tensor]): len(sequence_imgs) is equal to sequence_num,
                each element is a Tensor with shape of BxCxHxW.
        return:
            logits: output mask of size [B,C=3,H,W]
        """
        # encode cur_frame
        x1,x2,x3,x4,x5 = self.encoder(x)

        # encode mem_list
        x_mem = self.concat(mem_list) # B T C H W
        b,t,c,h,w = x_mem.shape
        x_mem = x_mem.reshape(b*t,c,h,w)
        _,_,_,_,x5_mem = self.encoder(x_mem)
        _,c_5,h_5,w_5 = x5_mem.shape
        x5_mem = x5_mem.reshape(b,t,c_5,h_5,w_5)
        x5_mem = torch.transpose(x5_mem,0,1) # T B C H W

        # fuse video features
        x_time = self.head(x5,x5_mem,y) #B C H W

        # decoder
        logits = self.decoder([x1,x2,x3,x4,x_time])
        return logits
    
    def concat(self,mem_list):
        for i,_ in enumerate(mem_list):
            mem_list[i] = mem_list[i].unsqueeze(1)
        x_mem = torch.cat(mem_list,dim=1) 
        return x_mem

