from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from torch import Tensor
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from ..utils import PAPPM, BasicBlock, Bottleneck
from mmseg.models.utils import resize


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 bias = False,
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2 if padding else 0,
            bias = bias, **kwargs)
        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x
    
class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride = 1,
                 padding=1,
                 bias = False,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2 if padding else 0, bias = bias,**kwargs)

        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=0.1)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x

def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    # print("x before mean and max:", x.shape)
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)[0]
    # mean_value = mean_value.unsqueeze(0)
    # print("mean max:", mean_value.shape, max_value.shape)

    if use_concat:
        res = torch.at([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res

def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            # print(xi.shape)
            res.extend(avg_max_reduce_channel_helper(xi, False))
        # print("res:\n",)
        # for it in res:
        #     print(it.shape)
        return torch.cat(res, dim=1)

class UMF(nn.Module):
    def __init__(self, in_ch, ksize=3, resize_mode='nearest'):
        super().__init__()
        self.conv_y = ConvBNReLU(
            in_ch, in_ch, kernel_size=ksize, padding=ksize // 2, bias=False)
        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                6, 6, kernel_size=3, padding=1, bias=False),
            ConvBN(
                6, 3, kernel_size=3, padding=1, bias=False))
        self.resize_mode = resize_mode
    
    def prepare(self, x, y, z):
        x = self.prepare_x(x, y, z)
        y = self.prepare_y(x, y, z)
        z = self.prepare_z(x, y, z)
        return x, y, z

    def prepare_x(self, x, y, z):
        return x

    def prepare_y(self, x, y, z):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        y_up = self.conv_y(y_up)
        return y_up
    
    def prepare_z(self, x, y, z):
        return z

    def fuse(self, x, y, z):
        atten = avg_max_reduce_channel([x, y, z])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        w1, w2, w3 = torch.split(atten, 1, 1)

        out = x * w1 + y * w2 + z * w3
        return out

    def forward(self, x, y, z):
        x, y, z = self.prepare(x, y, z)
        out = self.fuse(x, y, z)
        return out

class LUAFM(nn.Module):
    def __init__(self, x_ch, y_ch, ksize=3, resize_mode='nearest'):
        super().__init__()
        self.resize_mode = resize_mode
        self.x_ch = x_ch
        self.y_ch = y_ch
        self.min_ch = min(x_ch, y_ch)
        if x_ch>self.min_ch:
            self.conv_x = ConvBNReLU(
                x_ch, self.min_ch, kernel_size=ksize, padding=ksize // 2, bias=False)
        
        self.conv_y = ConvBNReLU(
            y_ch, self.min_ch, kernel_size=ksize, padding=ksize // 2, bias=False)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias=False),
            ConvBN(
                2, 1, kernel_size=3, padding=1, bias=False))
        
    def prepare_y(self, x, y):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        y_up = self.conv_y(y_up)
        return y_up
    def prepare_x(self, x, y):
        if self.x_ch > self.min_ch:
            x = self.conv_x(x)
        return x
    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))
        out = x * atten + y * (1 - atten)
        return out
    
    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out

@MODELS.register_module()
class BSSNet_T(BaseModule):
    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 ppm_channels: int = 96,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stem layer
        self.stem = self._make_stem_layer(in_channels, channels,
                                          num_stem_blocks)
        self.relu = nn.ReLU()

        # I Branch
        self.i_branch_layers = nn.ModuleList()
        for i in range(3):
            self.i_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2**(i + 1),
                    channels=channels * 8 if i > 0 else channels * 4,
                    num_blocks=num_branch_blocks if i < 2 else 2,
                    stride=2))

        # B Branch
        self.b_branch_layers = nn.ModuleList([
            ConvModule(
                channels*2,
                channels*2,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                channels*2,
                channels*4,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
        ])
        spp_module = PAPPM
        
        self.compression_b = ConvModule(
            channels*4, 
            channels*2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # D Branch
        self.d_branch_layers = nn.ModuleList([
            ConvModule(
                channels*2,
                channels*2,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                channels*2,
                channels*4,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
        ])

        self.compression_d = ConvModule(
            channels*4,
            channels*2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.spp = spp_module(
            channels * 16, ppm_channels, channels * 8, num_scales=5)
        
        self.uafm_ib = LUAFM(x_ch=channels*2, y_ch=channels*2)
        self.uafm_id = LUAFM(x_ch=channels*2, y_ch=channels*2)
        self.uafm4 = LUAFM(x_ch=channels*8, y_ch=channels*8)
        self.uafm3 = LUAFM(x_ch=channels*4, y_ch=channels*8)
        self.fuse_bdi = UMF(in_ch=channels*4)
        

    def _make_stem_layer(self, in_channels: int, channels: int,
                         num_blocks: int) -> nn.Sequential:
        """Make stem layer.

        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.

        Returns:

            nn.Sequential: The stem layer.
        """

        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.append(
            self._make_layer(BasicBlock, channels, channels, num_blocks))
        layers.append(nn.ReLU())
        layers.append(
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _make_layer(self,
                    block: BasicBlock,
                    in_channels: int,
                    channels: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Make layer for PIDNet backbone.
        Args:
            block (BasicBlock): Basic block.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Sequential: The Branch Layer.
        """
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)

        layers = [block(in_channels, channels, stride, downsample)]
        in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels,
                    channels,
                    stride=1,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))
        return nn.Sequential(*layers)

    def _make_single_layer(self,
                           block: Union[BasicBlock, Bottleneck],
                           in_channels: int,
                           channels: int,
                           stride: int = 1) -> nn.Module:
        """Make single layer for PIDNet backbone.
        Args:
            block (BasicBlock or Bottleneck): Basic block or Bottleneck.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Module
        """

        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
        return block(
            in_channels, channels, stride, downsample, act_cfg_out=None)

    def init_weights(self):
        """Initialize the weights in backbone.

        Since the D branch is not initialized by the pre-trained model, we
        initialize it with the same method as the ResNet.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(ckpt, strict=False)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        x = self.stem(x)
        x_b = self.b_branch_layers[0](x)
        x_d = self.d_branch_layers[0](x)
        x_i_3 = self.relu(self.i_branch_layers[0](x))
        comp_b = self.compression_b(x_i_3)
        x_b = self.uafm_ib(x_b, comp_b)
        comp_d = self.compression_d(x_i_3)
        x_d = self.uafm_id(x_d, comp_d)
        if self.training:
            tmp_xb = x_b.clone()
            tmp_xd = x_d.clone()
 
        x_b = self.b_branch_layers[1](x_b)
        x_d = self.d_branch_layers[1](x_d)
        x_i_4 = self.relu(self.i_branch_layers[1](x_i_3)) 

        x_i_5 = self.i_branch_layers[2](x_i_4)

        x_i_spp = self.spp(x_i_5)

        out = self.uafm4(x_i_4, x_i_spp) 
        out = self.uafm3(x_i_3, out) 
    
        out = self.fuse_bdi(x_b, out, x_d)
        return (tmp_xb, tmp_xd, x_i_3, x_i_4, x_i_5, out) if self.training else (out)

@MODELS.register_module()
class BSSNet_B(BaseModule):
    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 ppm_channels: int = 96,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stem layer
        self.stem = self._make_stem_layer(in_channels, channels,
                                          num_stem_blocks)
        self.relu = nn.ReLU()

        # I Branch
        self.i_branch_layers = nn.ModuleList()
        for i in range(3):
            self.i_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2**(i + 1),
                    channels=channels * 8 if i > 0 else channels * 4,
                    num_blocks=num_branch_blocks if i < 2 else 2,
                    stride=2))

        # B Branch
        self.b_branch_layers = nn.ModuleList([
            ConvModule(
                channels*2,
                channels*2,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                channels*2,
                channels*4,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
        ])
        spp_module = PAPPM
        
        self.compression_b = ConvModule(
            channels*4, 
            channels*2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # D Branch
        self.d_branch_layers = nn.ModuleList([
            ConvModule(
                channels*2,
                channels*2,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                channels*2,
                channels*4,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
        ])

        self.compression_d = ConvModule(
            channels*4,
            channels*2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.spp = spp_module(
            channels * 16, ppm_channels, channels * 8, num_scales=5)
        
        self.uafm_ib = LUAFM(x_ch=channels*2, y_ch=channels*2)
        self.uafm_id = LUAFM(x_ch=channels*2, y_ch=channels*2)
        self.uafm4 = LUAFM(x_ch=channels*8, y_ch=channels*8)
        self.uafm3 = LUAFM(x_ch=channels*4, y_ch=channels*8)
        self.fuse_bdi = UMF(in_ch=channels*4)
        
        self.upconv = ConvModule(
            channels*4,
            channels*4,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        

    def _make_stem_layer(self, in_channels: int, channels: int,
                         num_blocks: int) -> nn.Sequential:
        """Make stem layer.

        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.

        Returns:

            nn.Sequential: The stem layer.
        """

        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.append(
            self._make_layer(BasicBlock, channels, channels, num_blocks))
        layers.append(nn.ReLU())
        layers.append(
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _make_layer(self,
                    block: BasicBlock,
                    in_channels: int,
                    channels: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Make layer for PIDNet backbone.
        Args:
            block (BasicBlock): Basic block.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Sequential: The Branch Layer.
        """
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)

        layers = [block(in_channels, channels, stride, downsample)]
        in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels,
                    channels,
                    stride=1,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))
        return nn.Sequential(*layers)

    def _make_single_layer(self,
                           block: Union[BasicBlock, Bottleneck],
                           in_channels: int,
                           channels: int,
                           stride: int = 1) -> nn.Module:
        """Make single layer for PIDNet backbone.
        Args:
            block (BasicBlock or Bottleneck): Basic block or Bottleneck.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Module
        """

        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
        return block(
            in_channels, channels, stride, downsample, act_cfg_out=None)

    def init_weights(self):
        """Initialize the weights in backbone.

        Since the D branch is not initialized by the pre-trained model, we
        initialize it with the same method as the ResNet.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(ckpt, strict=False)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        x = self.stem(x)
        x_b = self.b_branch_layers[0](x)
        x_d = self.d_branch_layers[0](x)
        x_i_3 = self.relu(self.i_branch_layers[0](x))
        comp_b = self.compression_b(x_i_3)
        x_b = self.uafm_ib(x_b, comp_b)
        comp_d = self.compression_d(x_i_3)
        x_d = self.uafm_id(x_d, comp_d)
        if self.training:
            tmp_xb = x_b.clone()
            tmp_xd = x_d.clone()
        x_b = self.b_branch_layers[1](x_b)
        x_d = self.d_branch_layers[1](x_d)
        x_i_4 = self.relu(self.i_branch_layers[1](x_i_3)) 
        x_i_5 = self.i_branch_layers[2](x_i_4) 

        x_i_spp = self.spp(x_i_5)

        out = self.uafm4(x_i_4, x_i_spp)
        out = self.uafm3(x_i_3, out)

        out = self.fuse_bdi(x_b, out, x_d)
        out = self.upconv(out)
        out = resize(out, scale_factor=2)
        return (tmp_xb, tmp_xd, x_i_3, x_i_4, x_i_5, out) if self.training else (out)
    
    
    
    