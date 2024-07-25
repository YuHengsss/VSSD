
<div align="center">
<h1>VSSD </h1>
<h3>VSSD:  Vision Mamba with Non-Casual State Space Duality</h3>

[//]: # (Paper: &#40;[arXiv:2405.14174]&#40;https://arxiv.org/abs/2405.14174&#41;&#41;)
</div>

## Updates
* **` July. 25th, 2024`**: We release the code, log and ckpt for VSSD


## Introduction
This repository contains the code for training and evaluating VSSD varints on the ImageNet-1K dataset for image classification, COCO dataset for object detection, and ADE20K dataset for semantic segmentation.
For more information, please refer to our [paper]

[//]: # (&#40;https://arxiv.org/abs/2405.14174&#41;.)

<p align="center">
  <img src="./assets/overall_arc.jpg" width="800" />
</p>

## Main Results

### **Classification on ImageNet-1K**

|    name    | pretrain | resolution | acc@1 | #params | FLOPs |                                               logs&ckpts                                                | 
|:----------:| :---: | :---: |:-----:|:-------:|:-----:|:-------------------------------------------------------------------------------------------------------:| 
| VSSD-Micro | ImageNet-1K | 224x224 | 82.5  |   14M   | 2.3G  |   [log&ckpt](https://drive.google.com/drive/folders/1XWqLj4neH-MGktIe35l1orVUrKr6ry5V?usp=drive_link)   |
| VSSD-Tiny  | ImageNet-1K | 224x224 | 83.6  |   24M   | 4.5G  |                                              [log&ckpt](https://drive.google.com/drive/folders/1fguht9zoIBmS1WD9prqzYHD0APPG16Ub?usp=drive_link)                                              | 
| VSSD-Small | ImageNet-1K | 224x224 | 84.1  |   40M   | 7.4G  |                                              [log&ckpt](https://drive.google.com/drive/folders/1uXSfgD7A4ZVHRqcFFS7OQhbXIzVkoxB9?usp=drive_link)                                              | 
| VSSD-Base  | ImageNet-1K | 224x224 | 84.7  |   89M   | 16.1G |                                              [log&ckpt](https://drive.google.com/drive/folders/18KDn-jIi3NKnZ6e7Gd0-luEbbqQ1Q_6G?usp=drive_link)                                              | 

### **Object Detection on COCO**
  
|  Backbone  | #params | FLOPs | Detector | box mAP | mask mAP |     logs&ckpts     | 
|:----------:|:-------:|:-----:| :---: |:-------:|:--------:|:------------------:|
| VSSD-Micro |   33M   | 220G  | MaskRCNN@1x |  45.4   |   41.3   | [log&ckpt](https://drive.google.com/drive/folders/1yc_b0s4eE6iasEWIOSfiErIUew5747lf?usp=drive_link) |
| VSSD-Tiny  |   44M   | 265G  | MaskRCNN@1x |  46.9   |   42.6   | [log&ckpt](https://drive.google.com/drive/folders/1HZpm3s0gZnMb6Vh0WLqDzaFkp9-pSaXv?usp=drive_link) |
| VSSD-Small |   59M   | 325G  | MaskRCNN@1x |  48.4   |   43.5   | [log&ckpt](https://drive.google.com/drive/folders/1aBSa3hbHs7snNcQG_YY392GF9gINA2Io?usp=drive_link) |
| VSSD-Micro |   33M   | 220G  | MaskRCNN@3x |  47.7   |   42.8   | [log&ckpt](https://drive.google.com/drive/folders/1JIPfOIpYcKFbyeItiZBg5W2eXOzXXyXu?usp=drive_link) |
| VSSD-Tiny  |   44M   | 265G  | MaskRCNN@3x |  48.8   |   43.6   | [log&ckpt](https://drive.google.com/drive/folders/1ft17N0xme0gVmne6FISOoRZSuSWF42VF?usp=drive_link) |


### **Semantic Segmentation on ADE20K**

|   Backbone    | Input| #params | FLOPs | Segmentor | mIoU(SS) | mIoU(MS) |                                                                                          logs&ckpts                                                                                          |
|:-------------:| :---: |:-------:|:-----:| :---: |:--------:|:--------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|VSSD-Micro | 512x512 |   42M   | 893G  | UperNet@160k |   45.6   |   46.0   |                                             [log&ckpt](https://drive.google.com/drive/folders/1hJvpasGSFriz2IAci4nPLs9Su-sp8p3-?usp=drive_link)                                              | 
|  VSSD-Tiny   | 512x512 |   53M   | 941G  | UperNet@160k |   47.9   |   48.7   |                                             [log&ckpt](https://drive.google.com/drive/folders/1Jj8J0qAmuvKua4memX-eF-Ajd_ORJaUs?usp=drive_link)                                              | 


## Getting Started

### Installation

**Step 1: Clone the VSSD repository:**

```bash
git clone https://github.com/YuHengsss/VSSD.git
cd VSSD
```

**Step 2: Environment Setup:**

***Create and activate a new conda environment***

```bash
conda create -n VSSD
conda activate VSSD
```

[//]: # (***Install Dependencies***)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install -r requirements.txt)

[//]: # (cd kernels/selective_scan && pip install .)

[//]: # (```)

[//]: # (<!-- cd kernels/cross_scan && pip install . -->)


***Dependencies for `Detection` and `Segmentation` (optional)***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

<!-- conda create -n cu12 python=3.10 -y && conda activate cu12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# install cuda121 for windows
# install https://visualstudio.microsoft.com/visual-cpp-build-tools/
pip install timm==0.4.12 fvcore packaging -->


### Quick Start

**Classification**

To train VSSD models for classification on ImageNet, use the following commands for different configurations:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp
```

If you only want to test the performance (together with params and flops):

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp --resume </path/of/checkpoint> --eval
```

**Detection and Segmentation**

To evaluate with `mmdetection` or `mmsegmentation`:
```bash
bash ./tools/dist_test.sh </path/to/config> </path/to/checkpoint> 1
```
*use `--tta` to get the `mIoU(ms)` in segmentation*

To train with `mmdetection` or `mmsegmentation`:
```bash
bash ./tools/dist_train.sh </path/to/config> 8
```


[//]: # (## Citation)

[//]: # (If VSSD is helpful for your research, please cite the following paper:)

[//]: # (```)

[//]: # (@article{shi2024multiscale,)

[//]: # (      title={Multi-Scale VMamba: Hierarchy in Hierarchy Visual State Space Model}, )

[//]: # (      author={Yuheng Shi and Minjing Dong and Chang Xu},)

[//]: # (      journal={arXiv preprint arXiv:2405.14174},)

[//]: # (      year={2024})

[//]: # (})

[//]: # (```)

## Acknowledgment

This project is based on VMamba([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), Mambav2 ([paper](https://arxiv.org/abs/2405.21060), [code](https://github.com/state-spaces/mamba)), Swin-Transformer ([paper](https://arxiv.org/pdf/2103.14030.pdf), [code](https://github.com/microsoft/Swin-Transformer)), [OpenMMLab](https://github.com/open-mmlab),
 thanks for their excellent works.