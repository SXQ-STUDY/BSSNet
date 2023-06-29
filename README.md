# **BSSNet: A Real-Time Semantic Segmentation Network for Road Scenes Inspired from AutoEncoder**
**This is the official repository for our recent work: BSSNet**


## Overview
<p align="center">
  <img src="figs\overall.png" alt="overview-of-our-method" width="700"/></br>
</p>


## Highlights
<p align="center">
  <img src="figs\acc_speed.png" alt="overview-of-our-method" width="600"/></br>
</p>

- We propose a novel three-branch network for real-time segmentation, which extracts border, semantic, and spatial information separately.
- we are the first to introduce AutoEncoder into real-time semantic segmentation, which can extract spatial information in an unsupervised manner.
- We propose a Unified Multi-Feature Fusion module (UMF) that can efficiently fuse multiple features at a low computational cost and can be easily integrated into other models.

## Updates
   - 2023.06.08 Release Readme
   - 2023.06.29 Release the code

## Models

| Model (Cityscapes) | Val (% mIOU) | Test (% mIOU)| FPS |
|:-:|:-:|:-:|:-:|
| BSSNet-T | [79.0](https://drive.google.com/drive/folders/1aGz045inLcunQZfE8zl-N1JdCQFlxuOO?usp=drive_link) | [78.8](https://www.cityscapes-dataset.com/anonymous-results/?id=2b80f24f17f49d53b078768732e60d000220fd03ad056a713d9a6d6650c6c7eb) | 115.8 |
| BSSNet-B | [80.6](https://drive.google.com/drive/folders/1aGz045inLcunQZfE8zl-N1JdCQFlxuOO?usp=drive_link) | [80.5](https://www.cityscapes-dataset.com/anonymous-results/?id=8beaeef3b29f3dd6684a7d2b98a22d9586877ab2309dc5eebba814b95a46e0c8) | 39.2 |


| Model (CamVid) | Val (% mIOU) | Test (% mIOU)| FPS |
|:-:|:-:|:-:|:-:|
| BSSNet-T |-| [79.5](https://drive.google.com/drive/folders/1aGz045inLcunQZfE8zl-N1JdCQFlxuOO?usp=drive_link) | 170.8 |
| BSSNet-B |-| [81.6](https://drive.google.com/drive/folders/1aGz045inLcunQZfE8zl-N1JdCQFlxuOO?usp=drive_link) | 94.3 |

| Model (NightCity) | Val (% mIOU) | FPS |
|:-:|:-:|:-:|
| BSSNet-T| [52.6](https://drive.google.com/drive/folders/1aGz045inLcunQZfE8zl-N1JdCQFlxuOO?usp=drive_link) | 172.3 |
| BSSNet-B| [53.7](https://drive.google.com/drive/folders/1aGz045inLcunQZfE8zl-N1JdCQFlxuOO?usp=drive_link) | 117.2 |



## Prerequisites
This implementation is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Please refer to their repository for installation and dataset preparation. The inference speed is tested on single RTX 3090 using the method introduced by [SwiftNet](https://arxiv.org/pdf/1903.08469.pdf). No third-party acceleration lib is used.

## Usage
### 1. Prepare the dataset

* Download the [Cityscapes](https://www.cityscapes-dataset.com/), [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and [NightCity](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html) datasets.
* Add the dataset path to BSSNet_configs/_base_/datasets/xxdatasets.py

### 2. Training

````bash
python -m torch.distributed.launch --nproc_per_node=num_gpu tools/train.py configs_path --launcher pytorch
````

### 3. Evaluation

````bash
python tools/test.py config_path checkpoint_path
````


<!-- ## Citation

If you think this implementation is useful for your work, please cite our paper:
```
@misc{xu2022pidnet,
      title={PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller}, 
      author={Jiacong Xu and Zixiang Xiong and Shankar P. Bhattacharyya},
      year={2022},
      eprint={2206.02066},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->

## Acknowledgement

* Our implementation is modified based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
* Latency measurement code is borrowed from the [DDRNet](https://github.com/ydhongHIT/DDRNet).
* Thanks for their nice contribution.

