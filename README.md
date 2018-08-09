# Airbus Ship Detection Challenge
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/minerva-ml/open-solution-ship-detection/blob/master/LICENSE)

This is an open solution to the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection).

## Our goals
We are building entirely open solution to this competition. Specifically:
1. **Learning from the process** - updates about new ideas, code and experiments is the best way to learn data science. Our activity is especially useful for people who wants to enter the competition, but lack appropriate experience.
1. Encourage more Kagglers to start working on this competition.
1. Deliver open source solution with no strings attached. Code is available on our [GitHub repository :computer:](https://github.com/minerva-ml/open-solution-ship-detection). This solution should establish solid benchmark, as well as provide good base for your custom ideas and experiments. We care about clean code :smiley:
1. We are opening our experiments as well: everybody can have **live preview** on our experiments, parameters, code, etc. Check: [Airbus Ship Detection Challenge :chart_with_upwards_trend:](https://app.neptune.ml/neptune-ml/Ships?namedFilterId=mainListFilter) or screen below.

|Train and validation monitor :bar_chart:|
|:---:|
|[![training monitor](https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/02a2ab13edfe41cbad7e04c4a75b105393c14e02/ships_neptune.png)](https://app.neptune.ml/neptune-ml/Ships)|

## Disclaimer
In this open source solution you will find references to the [neptune.ml](https://neptune.ml). It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :snake:.

# How to start?
## Learn about our solutions
1. Check [Kaggle forum](https://www.kaggle.com/c/airbus-ship-detection/discussion/TODO) and participate in the discussions.
1. See solutions below:

| link to code | CV | LB |
|:---:|:---:|:---:|
|[solution 0](https://github.com/neptune-ml/open-solution-ship-detection/tree/solution-1)|XXX|XXX|
|solution 1|XXX|0.855|
|solution 2|XXX|0.868|

# Solution Write-up 
## Pipeline diagram

<img src="TODO"></img>

## Preprocessing
### :heavy_check_mark: What Worked
* Overlay binary masks for each image is produced 
* We load training and validation data in batches
* Only some basic augmentations (due to speed constraints) from the [imgaug package](https://github.com/aleju/imgaug) are applied to images 
* Image is resized before feeding it to the network

### :heavy_multiplication_x: What didn't Work

### :thinking: What could work
* Distances to the two closest objects are calculated creating the distance map that is used for weighing ([code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/master/src/preparation.py) :computer:).
* Size masks for each image is produced 
* Dropped small masks on the edges 
* Ground truth masks are prepared by first eroding them per mask creating non overlapping masks and only after that the distances are calculated 
* Ground truth masks for overlapping contours ([DSB-2018 winners](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741) approach).
* Dilated small objectcs to increase the signal 
* Network is fed with random crops


## Network
### :heavy_check_mark: What Worked
* Unet with Resnet152. This approach is explained in the [TernausNetV2](https://arxiv.org/abs/1806.00844) paper.
* Unet from scratch. Please take a look at our parametrizable [implementation of the U-Net](https://github.com/minerva-ml/steppy-toolkit/blob/master/toolkit/pytorch_transformers/architectures/unet.py#L9).

### :heavy_multiplication_x: What didn't Work

### :thinking: What could work
* unet with Resnet34, Resnet101 encoders
* linknets with Resnet34, Resnet101 Resnet152 encoders
* Unet with contextual blocks explained in [this paper](https://openreview.net/pdf?id=S1F-dpjjM).

## Loss function
### :heavy_check_mark: What Worked
* Using linear combination of soft dice and cross entropy

### :thinking: What could work
* Distance weighted cross entropy explained in the famous [U-Net paper](https://arxiv.org/pdf/1505.04597.pdf) (our [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/master/src/models.py#L227-L371) :computer: and [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/master/neptune.yaml#L79-L80) :bookmark_tabs:).
* Adding component weighted by building size (smaller buildings has greater weight) to the weighted cross entropy that penalizes misclassification on pixels belonging to the small objects ([code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/master/src/models.py#L227-L371) :computer:).
* Adding component that weighs in the importance of finding a ship (a lot of imbalance in this dataset)

## Training
### :heavy_check_mark: What Worked
* Use pretrained models!
* Sample images that contain ships

The entire configuration can be tweaked from the [config file](https://github.com/minerva-ml/open-solution-ship-detection/blob/master/configs/neptune.yaml) :bookmark_tabs:.

### :thinking: What could have worked but we haven't tried it
* Multistage training procedure along the lines of :
    1. train on a subset of the dataset with `lr=0.0001` and `dice_weight=0.5`
    1. train on a full dataset with `lr=0.0001` and `dice_weight=0.5`
    1. train with smaller `lr=0.00001` and `dice_weight=0.5`
    1. increase dice weight to `dice_weight=5.0` to make results smoother* Set different learning rates to different layers.
* Use cyclic optimizers.
* Use warm start optimizers.

## Postprocessing
### :heavy_check_mark: What Worked
* Simple morphological operations. At the beginning we used erosion followed by labeling and per label dilation with structure elements chosed by cross-validation. As the models got better, erosion was removed and very small dilation was the only one showing improvements ([code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/master/src/postprocessing.py) :computer:).

### :heavy_multiplication_x: What didn't Work

### :thinking: What could have work
* Test time augmentations by using colors
* Inference on reflection-padded or replication-padded images.
* Conditional Random Fields. It was so slow that we didn't check it for the best models 
* Test time augmentation (tta). Make predictions on image rotations (90-180-270 degrees) and flips (up-down, left-right) and take geometric mean on the predictions ([code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/master/src/loaders.py#L338-L497) :computer: and [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/master/src/pipeline_config.py#L119-L125) :bookmark_tabs:).
* Second level model.
* Ensembling
* Recurrent neural networks for postprocessing (instead of our current approach)

## Start experimenting with ready-to-use code
You can jump start your participation in the competition by using our starter pack. Installation instruction below will guide you through the setup.

### Installation *(fast track)*
1. Clone repository and install requirements (*use Python3.5*)
1. Register to the [neptune.ml](https://neptune.ml) _(if you wish to use it)_
1. Run experiment based on U-Net

:trident:
```bash
neptune account login
```

```bash
neptune run --config configs/neptune.yaml main.py prepare_masks
```

```bash
neptune run --config configs/neptune.yaml main.py prepare_metadata
```

```bash
neptune run --config configs/neptune.yaml main.py train --pipeline_name unet
```

```bash
neptune account login
neptune run --config configs/neptune.yaml main.py evaluate_predict --pipeline_name unet
```

:snake:
```bash
python main.py prepare_masks
```

```bash
python main.py prepare_metadata
```

```bash
python main.py -- train--pipeline_name unet
```

```bash
python main.py -- evaluate_predict --pipeline_name unet
```

## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/neptune-ml/open-solution-ship-detection/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in paticular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/airbus-ship-detection/discussion/TODO) is our primary way of communication.
1. Submit an [issue](https://github.com/minerva-ml/open-solution-ship-detection/issues) directly in this repo.