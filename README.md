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
1. Check [Kaggle forum](https://www.kaggle.com/c/airbus-ship-detection/discussion/62988) and participate in the discussions.
1. See solutions below:

| link to code | CV | LB |
|:---:|:---:|:---:|
|[solution 1](https://app.neptune.ml/neptune-ml/Ships?namedFilterId=1bc4da1e-6e47-4a26-a50e-3e55cbc052a7)|0.541|0.573|
|[solution 2](https://app.neptune.ml/neptune-ml/Ships?namedFilterId=8ad61fcb-f0ac-4aaf-aa9c-9db47e0aa222)|0.661|0.679|
|[solution 3](https://app.neptune.ml/neptune-ml/Ships?namedFilterId=be842434-7c8b-4ab9-afa5-f9c00816d3c3)|0.694|0.696|

## Start experimenting with ready-to-use code
You can jump start your participation in the competition by using our starter pack. Installation instruction below will guide you through the setup.

### Installation *(fast track)*
1. Clone repository and install requirements (*use Python3.5*) `pip3 install -r requirements.txt`
1. Register to the [neptune.ml](https://neptune.ml) _(if you wish to use it)_
1. Run experiment based on U-Net:


#### Cloud
```bash
neptune account login
```

Create project say Ships (SHIP)

Go to `neptune.yaml` and change:

```yaml
project: USERNAME/PROJECT_NAME
```
to your username and project name

Prepare metadata and overlayed target masks
It only needs to be **done once**

```bash
neptune send --worker xs \
--environment base-cpu-py3 \
--config neptune.yaml \
prepare_metadata.py

```

They will be saved in the

```yaml
  metadata_filepath: /output/metadata.csv
  masks_overlayed_dir: /output/masks_overlayed
```

From now on we will load the metadata by changing the `neptune.yaml`

```yaml
  metadata_filepath: /input/metadata.csv
  masks_overlayed_dir: /input/masks_overlayed
```

and adding the path to the experiment that generated metadata say SHIP-1 to every command `--input/metadata.csv`

Let's train the model by running the `main.py`:

```bash
neptune send --worker m-2p100 \
--environment pytorch-0.3.1-gpu-py3 \
--config neptune.yaml \
--input /SHIP-1/output/metadata.csv \
--input /SHIP-1/output/masks_overlayed \
main.py 

```

The model will be saved in the:

```yaml
  experiment_dir: /output/experiment
```

and the `submission.csv` will be saved in `/output/experiment/submission.csv`

You can easily use models trained during one experiment in other experiments.
For example when running evaluation we need to use the previous model folder in our experiment. We do that by:

changing `main.py` 

```python
  CLONE_EXPERIMENT_DIR_FROM = '/SHIP-2/output/experiment'
```

and running the following command:


```bash
neptune send --worker m-2p100 \
--environment pytorch-0.3.1-gpu-py3 \
--config neptune.yaml \
--input /SHIP-1/output/metadata.csv \
--input /SHIP-1/output/masks_overlayed \
--input /SHIP-2 \
main.py
```

#### Local
Login to neptune if you want to use it
```bash
neptune account login
```

Prepare metadata by running:

```bash
neptune run --config neptune.yaml prepare_metadata.py
```

Training and inference by running `main.py`:

```bash
neptune run --config neptune.yaml main.py
```

You can always run it with pure python :snake:

```bash
python main.py 
```

## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/neptune-ml/open-solution-ship-detection/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in particular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/airbus-ship-detection/discussion/62988) is our primary way of communication.
1. Submit an [issue](https://github.com/neptune-ml/open-solution-ship-detection/issues) directly in this repo.
