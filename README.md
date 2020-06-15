---
codeBlockCaptions: True
autoSectionLabels: True
linkReferences: True
link-citations: True
title: Deep learning for differentiation of steatohepatitis on MRI
---

<div id="wrapper">

<div id="header">

![](assets/perelman-logo.svg)

<div id="special">

<h1>Deep learning for differentiation of steatohepatitis on MRI</h1>

<p>
Ianto Lin Xi<sup>1</sup>;
Stephen J. Hunt, MD, PhD<sup>2</sup>
</p>

<div class="small">
1. Perelman School of Medicine, University of Pennsylvania, Pennsylvania, United States of America
2. Department of Radiology, Hospital of the University of Pennsylvania, Philadelphia, Pennsylvania, United States of America
</div>


</div>
</div>

<div id="columns">
<div class="column">

## Abstract

In this project, we will attempt to train a neural network to detect NASH on magnetic resonance imaging using retrospective MR imaging and labels derived from histopathology. A thorough literature review demonstrated no application of machine learning toward the diagnosis, differentiation, or staging of NASH. In this initial trial of deep learning methods, six different model architectures were tested. Preliminary results demonstrate the need for continued fine-tuning of these neural network architectures.

## Background

Advanced stage Non-Alcoholic Steatohepatitis (NASH) is currently the third most common indication for liver transplant with increasing incidence. Non-alcoholic Fatty Liver Disease (NAFLD) lies on a continuum of disease with NASH and is considered the most common liver disease in the world. Machine learning has been shown to perform at superhuman levels in detecting pathology on many different radiographic media[@mckinney2020international; @lakhani2017deep; @rajpurkar2017chexnet; @chi2017thyroid]. Applied clinically, an MRI-based diagnosis of NASH may allow patients to avoid painful and potentially morbid biopsies and may allow for better tracking and clinical decision making around usage of current and upcoming anti-fibrotic therapies.

## Methods

### Data collection, preparation, and augmentation

- T1, T1 with contrast, and T2WI studies and clinical information were collected from the UPHS PACS and EMR
- All studies and clinical data were anonymized and DICOM files were converted and saved as gzipped numpy arrays for faster access
- N4 bias correction was performed on all studies
- Studies were excluded based on the criteria in @fig:pipeline with final patient demographics summarized in @tbl:demographics
- Patients were split into training and validation sets at a ratio of 4:1
- During training, a single slice from 10%-20% of the depth of each volume was selected. This single slice was then cropped to 20%-60% of the width and height of the image
- Single cropped slices from T1, T1 with contrast, and T2 modalities were concatenated along the RGB channel. This is shown in @fig:dataset
- All training images were intensely augmented during training with shifts in hue, saturation, brightness, contrast, scale, rotation, sharpness, gaussian noise, and value. In addition, cutmix [@yun2019cutmix] was applied to 50% of training studies

</div><div class="column">

```{.mermaid format=svg loc=assets theme=neutral width=800 caption="Patient inclusion pipeline" #fig:pipeline}
graph TD
A[321 patients who had undergone transarterial liver biopsies since 2010]
B[106 patients had MRI studies available]
C[102 patients]
D[72 patients]
A --> | Search through PACS | B
B --> | Inclusion criteria: axial images, >20 slices, size > 100x100 | C
C --> | T1, T1C, and T2WI images available | D
```

### Model architecture

A number of different off-the-shelf backbone architectures were trialed including EfficientNet[@tan2019efficientnet], ResNet[@he2016deep], SE-ResNet[@hu2018squeeze], and Xception[@chollet2016xception]. All models were initialized with weights pretrained on ImageNet. Once features were extracted from these models, one dropout layer with 20% probability of drop out followed by one fully connected linear layer was used to output final class probabilities. Categorical cross entropy loss was used for gradient calculation during training. Training was performed with a mini-batch size of 32 studies, with gradient accumulation over two mini-batches. Learning rate was set to one tenth the learning rate of the lowest loss when the model is trained for 2 epochs with gradually increasing learning rate. Simple ensembles were created by averaging, taking the maximum, or taking the minimum probabilities of all models.

| Biopsy results                  | Steatohepatitis | No steatohepatitis |                 |
| :---------------------          | :------------   | :------------      | :-------------- |
| Patients                        | 37              | 35                 |                 |
| Age                             | 54.2 ± 13.3     | 55.4 ± 13.5        | p=0.70          |
| Males                           | 22              | 19                 | p=0.66          |
| Females                         | 15              | 15                 | p=0.84          |
|                                 |                 |                    |                 |
| **Comorbidities**               |                 |                    |                 |
| Viral hepatitis                 | 0               | 2                  | p=0.14          |
| Liver cancer history            | 1               | 3                  | p=0.28          |
| Diabetes                        | 10              | 6                  | p=0.32          |
| Hypertension                    | 9               | 7                  | p=0.66          |
|                                 |                 |                    |                 |
| **Other histological findings** |                 |                    |                 |
| Steatosis                       | 33              | 16                 | p<0.001         |
| Cirrhosis                       | 17              | 3                  | p<0.001         |
| Inflammation                    | 1               | 6                  | p=0.04          |

: Patient demographics {#tbl:demographics}

</div><div class="column">

![Sample images and pipeline](assets/stacked-pipeline.svg){#fig:dataset}

## Preliminary results

| Model              | Accuracy     | F1 score     | ROC AUC     | Precision     | Recall     | Specificity     |
| :----------------- | :----------- | :----------- | :---------- | :------------ | :--------- | :-------------- |
| Xception           | 0.53         | **0.66**     | 0.51        | 0.54          | **0.85**   | 0.16            |
| Efficientnet2_b0   | 0.48         | 0.57         | 0.47        | 0.51          | 0.64       | 0.3             |
| SE-Resnext50       | 0.55         | 0.53         | 0.56        | 0.6           | 0.48       | 0.64            |
| Resnet18           | 0.47         | 0.02         | 0.51        | **1**         | 0.01       | **1**           |
| Resnet34           | 0.4          | 0.34         | 0.41        | 0.42          | 0.28       | 0.54            |
| Resnet50           | **0.59**     | 0.55         | **0.6**     | 0.67          | 0.46       | 0.73            |
| **Ensembles**      |              |              |             |               |            |                 |
| Mean ensemble      | 0.45         | 0.06         | 0.49        | 0.38          | 0.03       | 0.94            |
| Max ensemble       | 0.45         | 0.25         | 0.47        | 0.46          | 0.17       | 0.78            |
| Min ensemble       | 0.46         | 0.08         | 0.5         | 0.5           | 0.04       | 0.95            |

: Performance of different model architectures {#tbl:performance}

![ROC curves of different model architectures](assets/figures/n4-xception.n4-efficientnet2_b0.n4-se_resnet50.n4-resnet18.n4-resnet34.n4-resnet50/roc-curve.svg){#fig:roc-models}

</div><div class="column">

## Discussion

This project aims to develop a machine learning model to detect steatohepatitis. From these early results, it is clear that an off-the-shelf approach is not sufficient. Next steps include:

- increasing the number of patients and images by obtaining or including data from:
    - saggital and coronal views
    - clinically diagnosed patients, to be used only in the training set.
    - other institutions for a multi-institutional dataset and greater generalizability.
- addressing the high correlation between different histological features with:
    - multilabel classification.
    - multitask learning.
    - regression labeling instead of classification.
    - customized class sampling strategies.
- leveraging the 3D structure of the data with 3D convolutional networks such as 3D ResNets[@tran2018closer]
- using additional data augmentation strategies such as:
    - 3D cutmix.
    - mixup/mixmatch[@eaton2018improving].
- incorporation of clinical history and laboratory data by:
    - amendment into final dense layers.
    - training a separate logistic regression model.
    - separating patients based on hyaluronic acid as a sign of cirrhosis[@pavlides2017multiparametric].
- improving our training methodology by:
    - setting aside a holdout test set.
    - incorporating k-fold cross validation.
    - experimenting with more sophisticated ensembling.
- attempting additional imaging modalities like ultrasound.

## Acknowledgements

This project is supported by the RSNA medical student research grant and the Department of Radiology at the Hospital of the University of Pennsylvania.

<h2>References</h2>
<div id="refs"></div>

</div>
</div>
</div>
