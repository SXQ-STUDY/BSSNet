**This is the project page for paper:[BSSNet: A Real-Time Semantic Segmentation Network for Road Scenes Inspired from AutoEncoder](https://ieeexplore.ieee.org/document/10286565)**


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
   - 2023.10.08 Accepted by TCSVT
   - 2023.10.13 Release the core code

## Experimental results

| Model (Cityscapes) | Val (% mIOU) | Test (% mIOU)| FPS |
|:-:|:-:|:-:|:-:|
| BSSNet-T | 79.0 | [78.8](https://www.cityscapes-dataset.com/anonymous-results/?id=2b80f24f17f49d53b078768732e60d000220fd03ad056a713d9a6d6650c6c7eb) | 115.8 |
| BSSNet-B | 80.6 | [80.5](https://www.cityscapes-dataset.com/anonymous-results/?id=8beaeef3b29f3dd6684a7d2b98a22d9586877ab2309dc5eebba814b95a46e0c8) | 39.2 |


| Model (CamVid) | Val (% mIOU) | Test (% mIOU)| FPS |
|:-:|:-:|:-:|:-:|
| BSSNet-T |-| 79.5 | 170.8 |
| BSSNet-B |-| 81.6 | 94.3 |

| Model (NightCity) | Val (% mIOU) | FPS |
|:-:|:-:|:-:|
| BSSNet-T| 52.6 | 172.3 |
| BSSNet-B| 53.7 | 117.2 |


## Getting Started
### Prerequisites
- This implementation is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Please refer to their repository for installation and dataset preparation. The inference speed is tested on single RTX 3090 using the method introduced by [SwiftNet](https://arxiv.org/pdf/1903.08469.pdf). No third-party acceleration lib is used.
- Download the [Cityscapes](https://www.cityscapes-dataset.com/), [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and [NightCity](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html) datasets. (Please email me if you want to use the processed dataset.)
- Add the dataset path to `BSSNet_configs/_base_/datasets/xx(dataset).py`


### Training
- Train BSSNet(e.g. on Cityscapes)
````bash 
python -m torch.distributed.launch --nproc_per_node=num_gpu tools/train.py BSSNet_configs\bssnet-cityscapes\bssnet-t-b12-120k-1024x1024-cityscapes.py --launcher pytorch
````

### Evaluation
- Evaluate BSSNet(e.g. on Cityscapes)
````bash
python tools/test.py BSSNet_configs\bssnet-cityscapes\bssnet-t-b12-120k-1024x1024-cityscapes.py checkpoint_path
````

### Train a custom dataset
- Adjust your dataset structure to the above supported dataset formats.

### Citation

If you think this implementation is useful for your work, please cite our paper:
```
@ARTICLE{10286565,
  author={Shi, Xiaoqiang and Yin, Zhenyu and Han, Guangjie and Liu, Wenzhuo and Qin, Li and Bi, Yuanguo and Li, Shurui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={BSSNet: A Real-Time Semantic Segmentation Network for Road Scenes Inspired From AutoEncoder}, 
  year={2024},
  volume={34},
  number={5},
  pages={3424-3438},
  keywords={Real-time systems;Semantics;Semantic segmentation;Feature extraction;Data mining;Computer architecture;Task analysis;Real-time semantic segmentation;convolution neural networks;AutoEncoder;feature fusion},
  doi={10.1109/TCSVT.2023.3325360}}
```

## Acknowledgement

* Our implementation is modified based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
* Latency measurement code is borrowed from the [DDRNet](https://github.com/ydhongHIT/DDRNet).
* Thanks for their nice contribution.

