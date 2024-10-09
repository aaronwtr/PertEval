<div align="center">

# PertEval: Evaluating Single-Cell Foundation Models for Perturbation Response Prediction

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.biorxiv.org/content/10.1101/2024.10.02.616248v1)
<!---
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)
--->
</div>

PertEval is a comprehensive evaluation framework designed for perturbation response
prediction.

Key features:

- **Extensive Model Support**: Evaluate a wide range of single-cell foundation models
  using simple probes for perturbation response prediction.
- **Standardized Evaluations**: Consistent benchmarking protocols and metrics for fair
  comparisons in transcriptomic perturbation prediction.
- **Flexible Integration**: Easily extend the codebase with custom models and datasets for
  perturbation prediction tasks.
- **Modular Design**: Built on top of PyTorch Lightning and Hydra, ensuring code
  organization and configurability.

PertEval-scFM is composed of three mains parts: data pre-processing, model training and
evaluation

![PertEval-scFM Graphical Abstract](figures/PertEval-scFM.png)

## Installation

<!---
#### Pip

{ADD PIP INSTALL PerturBench}
--->

To get PertEval up and running, first clone the GitHub repo:

```bash
# clone project
git clone https://github.com/aaronwtr/PertEval
cd PertEval
```

Set up a new conda or virtual environment and install the required dependencies:

```bash
# Conda
conda create -n perteval python=3.10
conda activate perteval

# Virtualenv
python3.10 -m venv perteval
### Windows:
perteval\Scripts\activate
### MacOS/Linux
source perteval/bin/activate
```

For a Windows install of torch with CUDA support, first run: 

```bash
# Windows
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
then run:

```bash
#Windows/MacOS/Linux
pip install -r requirements.txt
```

## Training

The main entry point for training or finetuning a model is `train.py`, which will load 
your data, model, and training configuration and start the training process.

You must choose your model configuration from the from [configs/experiment/](configs/experiment/) directory. 

```bash
python src/train.py experiment=experiment_name.yaml
```

We currently support three datasets:

* Norman_1
* Norman_2
* Replogle

and five single cell foundation models:

* scGPT
* Geneformer
* scfoundation
* scBERT
* UCE

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## Evaluation

To evaluate your model, run the following command with the path to the checkpoint file

```bash
python src/eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

## Evaluating on differentially expressed gene for a perturbation

Step 1) Calculate significant perturbations with
E-test [notebooks/preprocessing/significant_perts_edist.ipynb](notebooks/preprocessing/significant_perts_edist.ipynb)

Step 2) Calculate differentially expressed genes for all significant
perturbations [notebooks/preprocessing/diff_exp_refactored.ipynb](notebooks/preprocessing/diff_exp_refactored.ipynb)

Step 3) Prepare the inference
config [configs/experiment/mlp_norman_inference.yaml](configs/experiment/mlp_norman_inference.yaml).

- Add the path to the .ckpt file
- Add model you want to use
- Add the perturbation to be inspected
- Set the proper split and replicate corresponding to the perturbation. (TODO: Automate)
- Update the corresponding hidden and embedding dimensions (TODO: Automate)
- (Optional) Prepare shell scripts to run many perturbations at once on an
  HPC [notebooks/preprocessing/generate_deg_scripts.ipynb](notebooks/preprocessing/generate_deg_scripts.ipynb)

Step 4) Run eval.py with the inference config,

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

[//]: # (## Debugging)

[//]: # ()

[//]: # (The codebase is built on top of PyTorch Lightning and Hydra, which provides several useful features for debugging machine learning models. These features can help you quickly identify and fix issues during the training or fine-tuning of your model. Note that logs get placed into `logs/debugs/...`.)

[//]: # ()

[//]: # (#### Single Epoch Debugging)

[//]: # ()

[//]: # (When you want to quickly test your code or debug a specific issue, you can run the training for just a single epoch. This mode enforces debug-friendly configurations, such as setting all command-line loggers to debug mode, which provides more detailed logging output. To run training for a single epoch, use the following command:)

[//]: # ()

[//]: # (```bash)

[//]: # (python train.py debug=default)

[//]: # (```)

[//]: # ()

[//]: # (This mode is particularly useful when you want to verify that your data loading, model architecture, and training loop are working correctly before running the full training process.)

[//]: # ()

[//]: # (#### One Batch Debugging)

[//]: # ()

[//]: # (If you need to debug a specific batch or step in your training, validation, or testing loop, you can run the code for just one batch through each loop. This mode allows you to inspect the intermediate tensors, gradients, and other variables at each step, making it easier to identify and fix issues related to a specific batch or data sample. To run this mode, use the following command:)

[//]: # ()

[//]: # (```bash)

[//]: # (python train.py debug=fdr)

[//]: # (```)

[//]: # ()

[//]: # (This mode is particularly useful when you encounter issues with a specific batch or data sample, such as numerical instabilities, data loading errors, or model output inconsistencies.)

[//]: # ()

[//]: # (#### Overfitting to One Batch)

[//]: # ()

[//]: # (In some cases, you might want to test your model's ability to overfit to a single batch of data. This can be helpful for verifying that your model has enough capacity to learn the training data and for debugging issues related to the optimization process or loss function. To run this mode, use the following command:)

[//]: # ()

[//]: # (```bash)

[//]: # (python train.py debug=overfit)

[//]: # (```)

[//]: # ()

[//]: # (#### Detecting Numerical Anomalies)

[//]: # ()

[//]: # (To detect these anomalies in the model's tensors, you can enable the anomaly detection. This feature will check for NaNs or infinities in your tensors and raise an exception if any are found, helping you identify and fix the root cause of these numerical issues. To enable this feature, use the following command:)

[//]: # ()

[//]: # (```bash)

[//]: # (python train.py +trainer.detect_anomaly=true)

[//]: # (```)

[//]: # ()

[//]: # (This mode is particularly useful when you encounter numerical instabilities or unexpected behaviors during training or inference, as it can help you identify the source of these issues more quickly.)

#### Using a Subset of Data

In some cases, you might want to train or evaluate your model on a smaller subset of your
data, either for debugging purposes or to speed up the training process. PyTorch Lightning
provides options to limit the number of batches used for training, validation, and
testing. For example, to use only 20% of your data for each of these stages, you can run
the following command:

```bash
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

This mode can be useful when you want to quickly test your code or debug issues with a
smaller subset of your data, or when you want to perform a quick sanity check on your
model's performance before running the full training or evaluation process.


