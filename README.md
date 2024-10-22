# Invocab Configuration Setup and Execution Guide

This document provides detailed instructions to configure and run the invocab config. Follow the steps below to copy, rename, and launch the configuration.

## Steps to Follow

### 1. Copy the Configuration File

Start by copying the configuration file from the source path to the destination folder.

```bash
cp /tsi/hi-paris/GB/segmentation/configs/cityscapes_panoptic_train_invocab_008.json fc-clip/datasets/cityscapes/gtFine
```

### 2. Rename the File

Rename the copied configuration file to make it ready for use.

```bash
mv /home/ids/gbrison/segmentation/segmentation/fc-clip/datasets/cityscapes/gtFine/cityscapes_panoptic_train_invocab_008.json  fc-clip/datasets/cityscapes/gtFine/config.json
```


### 3. Launch the Configuration
Use the Python script to launch the configuration with the renamed file.

```bash
python train_net_GB.py --config-file  configs/coco/panoptic-segmentation/fcclip/r50_exp.yaml --num-gpus 2
```

### 4. Updating files 

-  fc-clip/train_net_GB.py
-  fc-clip/configs/coco/panoptic-segmentation/fcclip/r50_exp.yaml
-  fc-clip/fcclip/fcclip.py  "392 --- 410"
-  detectron2/detectron2/evaluation/panoptic_evaluation.py    
