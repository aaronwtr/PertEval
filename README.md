<div align="center">

# scBench: Benchmarking single-cell models across biological tasks 

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

scBench is a comprehensive benchmarking suite designed for evaluating single-cell foundation models and task-specific single-cell models. It offers out-of-the-box evaluations across a range of models and datasets, enabling researchers and developers to assess the performance of their models in a standardized manner.

The codebase is highly extensible, allowing seamless integration of custom models and datasets. You can incorporate your own models by creating a [LightningModule](#making-a-lightning-module), and introduce new datasets by implementing a [LightningDataModule](#making-a-lightning-datamodule).

Key features:

- **Extensive Model Support**: Evaluate a wide range of single-cell foundation models and task-specific models.
- **Standardized Evaluations**: Consistent benchmarking protocols and metrics for fair comparisons.
- **Flexible Integration**: Easily extend the codebase with custom models and datasets.
- **Modular Design**: Built on top of PyTorch Lightning and Hydra, ensuring code organization and configurability.

## Installation

#### Pip

{ADD PIP INSTALL SCBENCH}

Setting up conda environment:
```bash
# clone project
git clone https://github.com/aaronwtr/scBench
cd scBench

# [OPTIONAL] create conda or virtual environment
conda create -n scb python=3.10
conda activate scb
```

Alternatively, using virtualenv:
```
python3.10 -m venv scb

# Windows:
scb\Scripts\activate

# MacOS/Linux
source scb/bin/activate

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda
{TODO: implement conda install}
```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n scb

# activate conda environment
conda activate scb
```

## Making a Lightning DataModule 

[Lightning DataModule docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) 

{BODY EXPLANATION}

## Making a Lightning Module 

[Lightning Module docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)

{BODY EXPLANATION}

## How to run

The codebase has two entry points: `train.py` and `eval.py`. Which one you'll use depends on whether you want to train/fine-tune an existing model, or whether you have a pre-trained checkpoint you want to evaluate. 

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

Evals work similar by calling 

```bash
python src/eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

## Setting up Weights and Biases logging and experiment tracking
First install wandb via 

```bash
pip install wandb
```

and then login by pasting your API key when prompted via

```bash
wandb login
```

Then you can set the wandb project name and entity in the `configs/logger/wandb.yaml`.

```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

## Debugging

The codebase is built on top of PyTorch Lightning and Hydra, which provides several useful features for debugging machine learning models. These features can help you quickly identify and fix issues during the training or fine-tuning of your model. Note that logs get placed into `logs/debugs/...`.

#### Single Epoch Debugging

When you want to quickly test your code or debug a specific issue, you can run the training for just a single epoch. This mode enforces debug-friendly configurations, such as setting all command-line loggers to debug mode, which provides more detailed logging output. To run training for a single epoch, use the following command:

```bash
python train.py debug=default
```

This mode is particularly useful when you want to verify that your data loading, model architecture, and training loop are working correctly before running the full training process.

#### One Batch Debugging

If you need to debug a specific batch or step in your training, validation, or testing loop, you can run the code for just one batch through each loop. This mode allows you to inspect the intermediate tensors, gradients, and other variables at each step, making it easier to identify and fix issues related to a specific batch or data sample. To run this mode, use the following command:

```bash
python train.py debug=fdr
```

This mode is particularly useful when you encounter issues with a specific batch or data sample, such as numerical instabilities, data loading errors, or model output inconsistencies.

#### Overfitting to One Batch

In some cases, you might want to test your model's ability to overfit to a single batch of data. This can be helpful for verifying that your model has enough capacity to learn the training data and for debugging issues related to the optimization process or loss function. To run this mode, use the following command:

```bash
python train.py debug=overfit
```

#### Detecting Numerical Anomalies

To detect these anomalies in the model's tensors, you can enable the anomaly detection. This feature will check for NaNs or infinities in your tensors and raise an exception if any are found, helping you identify and fix the root cause of these numerical issues. To enable this feature, use the following command:

```bash
python train.py +trainer.detect_anomaly=true
```

This mode is particularly useful when you encounter numerical instabilities or unexpected behaviors during training or inference, as it can help you identify the source of these issues more quickly.

#### Using a Subset of Data

In some cases, you might want to train or evaluate your model on a smaller subset of your data, either for debugging purposes or to speed up the training process. PyTorch Lightning provides options to limit the number of batches used for training, validation, and testing. For example, to use only 20% of your data for each of these stages, you can run the following command:

```bash
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

This mode can be useful when you want to quickly test your code or debug issues with a smaller subset of your data, or when you want to perform a quick sanity check on your model's performance before running the full training or evaluation process.


