# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .bssnet_encoder_decoder import BSSNet_EncoderDecoder

__all__ = [
    'BaseSegmentor', 'BSSNet_EncoderDecoder'
]
