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
<img src="https://i.postimg.cc/1zqNsZH1/Screenshot-2024-08-27-at-21-27-24.png" alt="flow.png" width="1800px">
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

### Mathematical Formulation

###### Step 1: Extracting Losses from Two Models

**Model 1 Loss**

This model is specialized in out-of-vocabulary panoptic segmentation. Given the input data, the first model computes a set of losses:

`L1 = model_one(data) → loss_dict_model_one`

Here, `L1` is a dictionary containing individual loss components for Model 1.

**Model 2 Loss**

This model is specialized in in-vocabulary panoptic segmentation. Similarly, for the second model:

`L2 = model_two(data) → loss_dict_model_two`

`L2` is a dictionary containing individual loss components for Model 2.

###### Step 2: Combining the Losses into a New Dictionary

The losses from both models are combined into a single dictionary:

loss_dict = { 'm1 ' + i : L1[i] for each i ∈ L1 } ∪ { 'm2 ' + i : L2[i] for each i ∈ L2 }



###### Step 3: Computing the Combined Logits Using Softmax and Entropy

**Softmax Computation**

The softmax function is applied to the logits from both models:

$$
\text{Softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}
$$


where \(z_i\) is the i-th element of the input vector \(z\) (logits) and \(K\) is the total number of elements (classes).

For your models, the softmax outputs are:

`s1 = Softmax(logit_one[i])` 
`s2 = Softmax(logit_two[i])`


where `i` represents the key for logits (e.g., "pred logits", "pred masks").

**Entropy Calculation**

The entropy for each softmax output is calculated as:

$$
\text{Entropy}(s) = -\sum s \cdot \log(s + 1 \times 10^{-9})
$$

The small constant (1 * 10^(-9)) ensures numerical stability.

**New Logits**

The logits are combined using the calculated entropy to scale the softmax outputs:

`new_logit[i] = s1 * (1 - Entropy(s1)) + s2 * (1 - Entropy(s2))`

###### Step 4: Calculating the Combined Loss

The combined loss is computed using the criterion on the new logits:

Lcombined = criterion(new_logit, targets)

This loss is then added to the loss dictionary:

`loss_dict['com ' + i] = L_{combined}[i]` for each component `i`

###### Step 5: Summing and Backpropagation

Finally, the total loss to be backpropagated is computed by summing all individual losses:

`L_total = ∑ loss_dict[key]`

The total loss is then used for backpropagation:

Ltotal.backward()


###### Summary

The total loss function for the combined model training is a summation of the individual losses from two models along with a newly computed loss based on softmax and entropy. This approach allows leveraging the strengths of both models and introduces an additional regularization effect via entropy, promoting more confident predictions.
