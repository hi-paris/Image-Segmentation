# Image Segmentation In Vocab + Out Vocab 🚀

This readme is work in progress ⚙️

## 01 Installation 📦

➡️ [installation instructions](INSTALL.md)

## 02 Dataset 💽


<img src="https://i.postimg.cc/X7FPCzvD/0-If-TKQG09q-Kau-Kpba.png" alt="cityscape-dataset.png" width="400px">

Cityscape - Fine Annotations: These are high-quality annotations for 5,000 images collected in street scenes from 50 different cities, split into 2,975 for training, 500 for validation, and 1,525 for testing. The annotations include classes, such as road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle, and more.
<br />
Configuration 000 = all labels on 19 classes are presents. 11 classes (Stuff) & 8 classes (Things)
Configuration 008 = Only labels on Stuff classes are presents labels on things are removed 
<br />
| Conf-ID | deleted classes | 
|---------|------------------|
| 000     | \{-\}            |
| 001     | \{24\}           |
| 002     | \{24, 25\}       |
| 003     | \{24, 25, 26\}   |
| 004     | \{24, 25, 26, 27\}|
| 005     | \{24, 25, 26, 27, 28\}|
| 006     | \{24, 25, 26, 27, 28, 31\}|
| 007     | \{24, 25, 26, 27, 28, 31, 32\}|
| 008     | \{24, 25, 26, 27, 28, 31, 32, 33\}|

<br />

train: 

<br />

<img src="https://i.postimg.cc/c4Jnmxbv/frequency-of-each-class.png" alt="frequency-of-each-class.png" width="300px">

<br />

val: 

<br />

<img src="https://i.postimg.cc/MpCqv5BW/frequency-of-each-class-val.png" alt="frequency-of-each-class-val.png" width="300px">



## 04 Approaches ➡️

The goal was to find an innovative approach for panoptic segmentation that works for In and Out of vocabulary in finetuning process

Our current top performer architecture: >>
<img src="https://i.postimg.cc/mrNr2Z0M/Screenshot-2024-08-27-at-21-22-23.png" alt="flow.png" width="1500px">
<br />

**Generate Configurations** 

➡️ [Configurations Generation](data-generation/README.md)

**Zero Shot (FCCLIP) - 000 - inference**

1️⃣ Step 1: 

rename fcclip/fc-clip/fcclip_inference >> fcclip/fc-clip/fcclip

2️⃣ Step 2: 

```bash
python train_net_inference.py --config-file /home/infres/gbrison/fc3/fc-clip/configs/coco/panoptic-segmentation/fcclip/r50_exp.yaml --eval-only MODEL.WEIGHTS /home/infres/gbrison/fc3/fc-clip/fcclip_cocopan_r50.pth
```

**Upperbound - FT - 000 - 1000 iter**

1️⃣ Step 1: 

rename fcclip/fc-clip/fcclip_normal >> fcclip/fc-clip/fcclip

2️⃣ Step 2: 

```bash
python train_net_normal.py --config-file /home/infres/gbrison/fc3/fc-clip/configs/coco/panoptic-segmentation/fcclip/r50_exp.yaml --eval-only MODEL.WEIGHTS /home/infres/gbrison/fc3/fc-clip/fcclip_cocopan_r50.pth
```

**FTZS=FT on in-classes (Ground Truth) + ZS predictions for out  - 008 - 1000 iter**

1️⃣ Step 1: 

rename fcclip/fc-clip/fcclip_invocab >> fcclip/fc-clip/fcclip

2️⃣ Step 2: 

```bash
python train_net_invocab.py --config-file /home/infres/gbrison/fc3/fc-clip/configs/coco/panoptic-segmentation/fcclip/r50_exp.yaml --eval-only MODEL.WEIGHTS /home/infres/gbrison/fc3/fc-clip/fcclip_cocopan_r50.pth
```


**FT on in-classes-GT (out -> void) + intersection with ZS at inference  - 008 - 1000 iter**

1️⃣ Step 1: 

rename fcclip/fc-clip/fcclip_naive >> fcclip/fc-clip/fcclip

2️⃣ Step 2: 

```bash
python train_net_naive.py --config-file /home/infres/gbrison/fc3/fc-clip/configs/coco/panoptic-segmentation/fcclip/r50_exp.yaml --num-gpus 2
```

**Knowledge Distillation (Teacher -Student)**

1️⃣ Step 1: 

rename fcclip/fc-clip/fcclip_kd >> fcclip/fc-clip/fcclip

2️⃣ Step 2: 

```bash
python train_net_kd.py --config-file /home/infres/gbrison/fc3/fc-clip/configs/coco/panoptic-segmentation/fcclip/r50_exp.yaml --num-gpus 2
```

** Combination of Models ** 



## 03 Results ✨



| Cityscapes FCCLIP Approaches                                                                                           |  Perf All (PQ) | Perf Stuff/In (PQ) | Perf Things/Out (PQ) |
|-------------------------------------------------------------------------------------------------------------------------|--------|----------|------------|
| **Upperbound: Full finetuning - 000 - 1000 iter - Resnet 50**                                                               | **58.197** | **63.995**   | **50.224**     |
| FT on in-classes-GT (out -> void) - 1000 iter + intersection with ZS at inference (Naive) - Resnet 50                   | 50.35  | 65.575   | 29.416     |
| **FTZS=FT on in-classes (Ground Truth) + ZS predictions for out  - 008 - 1000 iter - Resnet 50 (Invocab)**                  | **52.972** | **58.519**   | **45.346**     |
| Zero Shot (FCCLIP) 000 - Resnet 50                                                                                      | 40.311 | 48.234   | 29.416     |
| FT on in-classes-GT (out -> void)- 100 iter (train normal 008) - Resnet 50                                              | 35.133 | 60.684   | 0          |
| **Lower bound: FT on in-classes-GT (out -> void)- 1000 iter (train normal 008) - Resnet 50**                                | **37.965** | **65.575**   | **0**          |
| Intersection n°5 + n°9                                                                                                  | 50.41  | 65.575   | 29.416     |
| Intersection n°6 + n°9                                                                                                  | 56.935 | 65.575   | 44.974     |
| n°6 + n°9 - Combined FT - L1 + L2 + Lcombined (mean of Logits of sum of M1+M2) - Pred N°6 - 1000 iter                   | 56.902 | 63.457   | 47.889     |
| n°9 + n°6 - Combined FT - L1 + L2 + Lcombined (mean of the sum of Logits of M1+M2) - Pred N°9 - 1000 iter               | 55.164 | 64.534   | 42.28      |
| n°6 + n°9 - Combined FT - L1 + L2 + Lcombined (weighted of Logits 0.6 0.4) - 1000 iter                                  | 57.422 | 64.234   | 48.056     |
| n°6 + n°9 - Combined FT - L1 + L2 + Lcombined softmax - sum - threshold 0.7 - 1000 iter                                 | 57.745 | 63.719   | 49.531     |
| n°6 + n°9 - Combined FT - L1 + L2 + Lcombined 2 softmax - sum - top 10 - 1000 iter                                      | 57.932 | 63.929   | 49.688     |
| **n°6 + n°9 - Combined FT - L1 + L2 + Lcombined 2 softmax - entropy - 1000 iter**                                           | **58.227** | **64.238**   | **49.962**     |
| Zero Shot (FCCLIP) 000 - ConvnextLarge                                                                                  | 43.992 | 49.662   | 36.195     |
| FT on in-classes-GT (out -> void)- 1000 iter (train normal 008) - ConvNextLarge                                         | 33.231 | 57.398   | 0          |
| FT on in-classes-GT (out -> void)- 1000 iter (train normal 008) - ConvNextLarge                                         | 34.052 | 58.818   | 0          |
| FT on in-classes-GT (out -> void) - 1000 iter + intersection with ZS at inference (Naive) - ConvNextLarge               |        |          |            |




--------------
<br />

ℹ️ There is an extra class that is belonging to the void

* Upper bound is fine-tuning of FCCLIP on 1000 iterations - configuration 000

* Lower bound is fine-tuning of FCCLIP on 1000 iterations - configuration 008

## 04 The maths behind it 🧮

<br />

**PQ Metric**

The Panoptic Quality (PQ) is calculated as follows:

PQ = $\frac{\sum_{(p, g) \in TP} \text{IoU}(p, g)}{TP + \frac{1}{2} \times (FP + FN)}$

Where:
- TP (True Positives) refers to the set of correctly matched predicted segments to the ground truth.
- IoU(p, g) represents the Intersection over Union for the predicted segment p and the ground truth segment g.
- FP (False Positives) are the predicted segments with no corresponding ground truth.
- FN (False Negatives) are the ground truth segments that were not predicted.

----
<br />

**Softmax**

The Softmax function is defined as follows for a vector **z** of real numbers and its ith component:

Softmax($z_i$) = $\frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$

Where:
- $z_i$ is the score or logit for the ith class.
- $e^{z_i}$ is the exponential of $z_i$.
- $\sum_{j=1}^{n} e^{z_j}$ is the sum of exponentials of all components of the vector **z**, which serves as the normalization term to ensure the outputs sum up to one, thus forming a probability distribution.

----
<br />

**Cross-entropy**

Cross-entropy loss is used to measure the dissimilarity between the true probability distribution $p$ and the predicted distribution $q$, for a given set of classes. It is defined as follows:

Cross Entropy Loss = $-\sum_{i=1}^{n} p_i \log(q_i)$

Where:
- $p_i$ is the true probability of class $i$, often represented as 1 for the true class and 0 for others in classification tasks.
- $q_i$ is the predicted probability of class $i$, as output by a model, such as from a softmax function.
- $n$ is the number of classes.
- $\log(q_i)$ represents the natural logarithm of the predicted probability $q_i$.

----
