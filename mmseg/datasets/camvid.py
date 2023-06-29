# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CamVidDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('sky', 'building', 'column_pole', 'road', 'slidewalk', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist'),
        palette=[[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192]]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
