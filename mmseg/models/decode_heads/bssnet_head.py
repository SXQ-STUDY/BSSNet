from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList, ConfigType


class BaseBSSNet_Head(BaseModule):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv'))
        _, self.norm = build_norm_layer(norm_cfg, num_features=channels)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor, cls_seg: Optional[nn.Module]) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if cls_seg is not None:
            x = cls_seg(x)
        return x

@MODELS.register_module()
class BSSNet_Head(BaseDecodeHead):
    def __init__(self,
                 in_channels: list,
                 channels: int,
                 num_classes: int,
                 loss_detail_weight=1.0,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.loss_detail_weight = loss_detail_weight
        
        self.fuse_b_d_head = BaseBSSNet_Head(self.in_channels[2], channels, norm_cfg, act_cfg)

        if self.training:
            self.b_head = BaseBSSNet_Head(
                self.in_channels[0],
                self.in_channels[0] // 2,
                norm_cfg,
            )
            self.b_cls_seg = nn.Conv2d(self.in_channels[0] // 2, 1, kernel_size=1)

            self.d_head = BaseBSSNet_Head(
                self.in_channels[1],
                self.in_channels[1],
                norm_cfg=norm_cfg
            )
            self.d_cls_seg = nn.Conv2d(self.in_channels[1], 3, kernel_size=1)

        self.loss_detail = nn.L1Loss(size_average=True, reduce=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def loss(self, backbone_feas, inputs, batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        seg_logits = self.forward(backbone_feas)
        losses = self.loss_by_feat(seg_logits, inputs, batch_data_samples)
        return losses

    def forward(
            self,
            inputs: Union[Tensor,
                          Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            x = self._transform_inputs(inputs)
            assert len(x) == 3
            x_b, x_d, fuse_b_d = x
            fuse_b_d = self.fuse_b_d_head(fuse_b_d, self.cls_seg)
            x_b = self.b_head(x_b, self.b_cls_seg)
            x_d = self.d_head(x_d, self.d_cls_seg)
            return fuse_b_d, x_b, x_d
        else:
            return self.fuse_b_d_head(inputs, self.cls_seg)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = [
            data_sample.gt_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     inputs,
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        sem_i_logit, b_logit, d_logit = seg_logits
        sem_label, bd_label = self._stack_batch_gt(batch_data_samples)

        sem_i_logit = resize(
            input=sem_i_logit,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        b_logit = resize(
            input=b_logit,
            size=bd_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        d_logit = resize(
            input=d_logit,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        sem_label = sem_label.squeeze(1)
        bd_label = bd_label.squeeze(1)

        loss['loss_sem_i'] = self.loss_decode[0](sem_i_logit, sem_label)
        loss['loss_bd'] = self.loss_decode[1](b_logit, bd_label)
        filler = torch.ones_like(sem_label) * self.ignore_index
        sem_bd_label = torch.where(
            torch.sigmoid(b_logit[:, 0, :, :]) > 0.8, sem_label, filler)
        loss['loss_sem_bd'] = self.loss_decode[2](sem_i_logit, sem_bd_label)
        loss['loss_detail'] = self.loss_detail(d_logit, inputs) * self.loss_detail_weight

        loss['acc_seg'] = accuracy(
            sem_i_logit, sem_label, ignore_index=self.ignore_index)
        return loss
