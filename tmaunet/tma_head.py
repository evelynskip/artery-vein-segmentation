import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from mmcv.cnn import ConvModule
# from .decode_head import BaseDecodeHead
from .custom import SequenceConv
import time


class MemoryModule(nn.Module):
    """Memory read module.
    Args:

    """

    def __init__(self,
                 matmul_norm=False):
        super(MemoryModule, self).__init__()
        self.matmul_norm = matmul_norm

    def forward(self, memory_keys, memory_values, query_key, query_value):
        """
        Memory Module forward.
        Args:
            memory_keys (Tensor): memory keys tensor, shape: TxBxCxHxW
            memory_values (Tensor): memory values tensor, shape: TxBxCxHxW
            query_key (Tensor): query keys tensor, shape: BxCxHxW
            query_value (Tensor): query values tensor, shape: BxCxHxW

        Returns:
            Concat query and memory tensor.
        """
        sequence_num, batch_size, key_channels, height, width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_key.shape[1] == key_channels and query_value.shape[1] == value_channels
        memory_keys = memory_keys.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys = memory_keys.view(batch_size, key_channels, sequence_num * height * width)  # BxCxT*H*W

        query_key = query_key.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
        key_attention = torch.bmm(query_key, memory_keys)  # BxH*WxT*H*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=-1)  # BxH*WxT*H*W

        memory_values = memory_values.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_values = memory_values.view(batch_size, value_channels, sequence_num * height * width)
        memory_values = memory_values.permute(0, 2, 1).contiguous()  # BxT*H*WxC
        memory = torch.bmm(key_attention, memory_values)  # BxH*WxC
        memory = memory.permute(0, 2, 1).contiguous()  # BxCxH*W
        memory = memory.view(batch_size, value_channels, height, width)  # BxCxHxW

        query_memory = torch.cat([query_value, memory], dim=1)
        return query_memory


class TMAHead(nn.Module):
    """TMAHead decoder for video semantic segmentation."""
    def __init__(self, in_channels, out_channels,sequence_num, key_channels, value_channels):
        """
        Args:
            in_channels: the channel num of backbone outputs
            sequence_num: the number of reference frames
            key_channels: the channel num of keys
            value_channels: the channel num of values
        """
        super(TMAHead, self).__init__()
        self.conv_cfg = None
        self.norm_cfg = dict(type='SyncBN', requires_grad=True)
        self.act_cfg = dict(type='ReLU')
        self.sequence_num = sequence_num
        self.in_channels = in_channels
        self.channels = out_channels
        self.memory_key_conv = nn.Sequential(
            SequenceConv(self.in_channels, key_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(key_channels, key_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        self.memory_value_conv = nn.Sequential(
            SequenceConv(self.in_channels, value_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(value_channels, value_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        self.query_key_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )

        self.query_value_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        self.memory_module = MemoryModule(matmul_norm=False)
        self.bottleneck = ConvModule(
            value_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def forward(self, inputs, sequence_imgs):
        """
        Forward fuction.
        Args:
            inputs ([Tensor]): backbone last-layer outputs. shape BxCxHxW
            sequence_imgs ([Tensor]): a Tensor with shape of TxBxCxHxW.
            inputs (list[Tensor]): backbone multi-level outputs.
            sequence_imgs (list[Tensor]): len(sequence_imgs) is equal to batch_size,
                each element is a Tensor with shape of TxCxHxW.
        Returns:
            output([Tensor]): fused features with shape B C H W 
        """
        x = inputs
        sequence_num, batch_size, channels, height, width = sequence_imgs.shape

        assert sequence_num == self.sequence_num
        memory_keys = self.memory_key_conv(sequence_imgs)
        memory_values = self.memory_value_conv(sequence_imgs)
        query_key = self.query_key_conv(x)  # BxCxHxW
        query_value = self.query_value_conv(x)  # BxCxHxW

        # memory read
        output = self.memory_module(memory_keys, memory_values, query_key, query_value)
        output = self.bottleneck(output)
        return output


class MemoryModuleForce(nn.Module):
    """Memory read module.
    Args:

    """

    def __init__(self,
                 matmul_norm=False):
        super(MemoryModuleForce, self).__init__()
        self.matmul_norm = matmul_norm

    def forward(self, memory_keys, memory_values, query_key, query_value,force_value):
        """
        Memory Module forward.
        Args:
            memory_keys (Tensor): memory keys tensor, shape: TxBxCxHxW
            memory_values (Tensor): memory values tensor, shape: TxBxCxHxW
            query_key (Tensor): query keys tensor, shape: BxCxHxW
            query_value (Tensor): query values tensor, shape: BxCxHxW
            force_value (List): shape: BxTx6

        Returns:
            Concat query and memory tensor.
        """
        sequence_num, batch_size, key_channels, height, width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_key.shape[1] == key_channels and query_value.shape[1] == value_channels
        memory_keys = memory_keys.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys = memory_keys.view(batch_size, key_channels, sequence_num * height * width)  # BxCxT*H*W

        query_key = query_key.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
        key_attention = torch.bmm(query_key, memory_keys)  # BxH*WxT*H*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=-1)  # BxH*WxT*H*W

        memory_values = memory_values.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        # calculate weights F:[min max cur]
        D_F_min = force_value[:,0,2]-force_value[:,2,2]
        D_min_max = force_value[:,0,2]-force_value[:,1,2]
        D_F_max = force_value[:,2,2]-force_value[:,1,2]
      
        w_min = D_F_min/D_min_max # BxT  D to F_max
        w_max = D_F_max/D_min_max # D to F_min

        weights = torch.stack((w_min, w_max), dim=1).unsqueeze(1).unsqueeze(3).unsqueeze(4)# T[min/max]
        memory_values = memory_values*weights # B C T H W

        memory_values = memory_values.view(batch_size, value_channels, sequence_num * height * width)
        memory_values = memory_values.permute(0, 2, 1).contiguous()  # BxT*H*WxC
        memory = torch.bmm(key_attention, memory_values)  # BxH*WxC
        memory = memory.permute(0, 2, 1).contiguous()  # BxCxH*W
        memory = memory.view(batch_size, value_channels, height, width)  # BxCxHxW

        query_memory = torch.cat([query_value, memory], dim=1)
        return query_memory


class TMAHeadForce(nn.Module):
    """TMAHead decoder for video semantic segmentation.Use forces to change weights"""
    def __init__(self, in_channels, out_channels,sequence_num, key_channels, value_channels):
        """
        Args:
            in_channels: the channel num of backbone outputs
            sequence_num: the number of reference frames
            key_channels: the channel num of keys
            value_channels: the channel num of values
        """
        super(TMAHeadForce, self).__init__()
        self.conv_cfg = None
        self.norm_cfg = dict(type='SyncBN', requires_grad=True)
        self.act_cfg = dict(type='ReLU')
        self.sequence_num = sequence_num
        self.in_channels = in_channels
        self.channels = out_channels
        self.memory_key_conv = nn.Sequential(
            SequenceConv(self.in_channels, key_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(key_channels, key_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        self.memory_value_conv = nn.Sequential(
            SequenceConv(self.in_channels, value_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(value_channels, value_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        self.query_key_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )

        self.query_value_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        self.memory_module = MemoryModuleForce(matmul_norm=False)
        self.bottleneck = ConvModule(
            value_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def forward(self, inputs, sequence_imgs,force_value):
        """
        Forward fuction.
        Args:
            inputs ([Tensor]): backbone last-layer outputs. shape BxCxHxW
            sequence_imgs ([Tensor]): a Tensor with shape of TxBxCxHxW.
            inputs (list[Tensor]): backbone multi-level outputs.
            sequence_imgs (list[Tensor]): len(sequence_imgs) is equal to batch_size,
                each element is a Tensor with shape of TxCxHxW.
            force_value([Tensor]): BxLx6 L=2/3 (min) (max) (cur)
        Returns:
            output([Tensor]): fused features with shape B C H W 
        """
        x = inputs
        # sequence_imgs = [self._transform_inputs(inputs).unsqueeze(0) for inputs in sequence_imgs]  # T, BxCxHxW
        # sequence_imgs = torch.cat(sequence_imgs, dim=0)  # TxBxCxHxW
        sequence_num, batch_size, channels, height, width = sequence_imgs.shape

        assert sequence_num == self.sequence_num
        memory_keys = self.memory_key_conv(sequence_imgs)
        memory_values = self.memory_value_conv(sequence_imgs)
        query_key = self.query_key_conv(x)  # BxCxHxW
        query_value = self.query_value_conv(x)  # BxCxHxW

        # memory read
        output = self.memory_module(memory_keys, memory_values, query_key, query_value,force_value)
        output = self.bottleneck(output)
        return output
    

def concat(mem_list):
        for i,_ in enumerate(mem_list):
            mem_list[i] = mem_list[i].unsqueeze(1)
        x_mem = torch.cat(mem_list,dim=1) 
        return x_mem