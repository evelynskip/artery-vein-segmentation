# artery-vein-segmentation
Implementation of Article ["Force Sensing Guided Artery-Vein Segmentation via Sequential Ultrasound Images"](https://arxiv.org/abs/2407.21394)

## Install
    pip install -r requirements-gpu.txt 
## Generate Dataset
    python utils/generate_dataset.py
## Train
    python train.py -n fg-unet -e 100 -b 4 -l 6e-6 -g 5 -s 2 

## Test
    python demo_dir.py -n fg-unet -m checkpoints/fg-unet_E_60_G5_LR6e-06_BS_4_S_2.pth

## Ablation Experiments
    python train.py -n unet -e 100 -b 4 -l 6e-6 -g 5 -s 1 
    python train.py -n seq-unet -e 100 -b 4 -l 6e-6 -g 5 -s 2 

## Reference
[TMANet](https://github.com/wanghao9610/TMANet)

[UNet](https://github.com/milesial/Pytorch-UNet)