<div align="center">

# PertEval: Evaluating Single-Cell Foundation Models for Perturbation Response Prediction

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
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
You'll need to create a Weights and Biases account if you don't have one already, and then login by pasting your API key when prompted
```bash
wandb login <key here>
```
 
alternatively, you can log in via command line before starting the training process.  

## Single-cell Foundation Models

We currently host embeddings of the Norman _et al._, 2019 labeled Perturb-seq dataset 
for 1 gene (`Norman_1`) and 2 gene perturbations (`Norman_2`) 
extracted with the following single-cell foundation models (scFM):

| **Model name** | **Architecture**   | **Pre-training objective** | **# of cells**   | **Organism** | **Emb. dim.** |
|-------------------|----------------|--------------------------|------------------|--------------|--------------|
| **scBERT**            | Performer      | Masked language modeling | ~5 million  | human  mouse | 200 |
| **Geneformer**        | Transformer    | Masked language modeling | ~30 million | human        | 256          |
| **scGPT**             | FlashAttention | Specialized attention-masking mechanism | ~33 million | human        | 512          |
| **UCE**               | Transformer    | Masked language modeling | ~36 million | 8 species    | 1,280        |
| **scFoundation**      | Transformer    | Read-depth-aware modeling | ~50 million | human        | 3,072        |

### Embeddings

The control expression data and scFM embeddings will be automatically 
downloaded, stored and preprocessed during the initial training run. The 
embeddings will be stored in the `/data/splits/perturb/norman_x/embeddings` directory.

## Training and Evaluation

The main entry point for training and validation of a model is `train.py`, 
which will load your data, model, configs and run the training and validation process. 
`eval.py` will evaluate a trained model on the test set. You can run training and 
testing using the best checkpoints for the run by setting both `train` and `test` to 
`True` in `train.yaml`. 

To run a specific experiment, point to the corresponding configuration file from the
[configs/experiment/](configs/experiment/) directory. For example:

```bash
python src/train.py experiment=mlp_norman_train.yaml
```

In the config file you will be able to modify the configuration file to suit 
your needs, such as batch size, learning rate, or number of epochs. You can also 
override these parameters from the command line. For example:

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

For **Norman_1** the input size is **2060**, and for **Norman_2** the input size 
is **2064**. The embedding dimension of each scFM can be found in the table above. 
The hidden layer is embed. dim. / 2.

### Modeling distribution shift with SPECTRA

We model distribution shift by creating increasingly challenging train-test splits with 
SPECTRA, a graph-based method which controls for cross-split overlap between 
train-test data. The splits are created during the initial training run and stored in
`/data/splits/perturb/norman_x/norman_x_SPECTRA_splits` directory. The sparsification 
probability (_s_) controls the connectivity in the sample-to-sample similarity graph. To 
assess distribution shift, you will have to train and test the model on the different 
values of _s_ in the config file `split: 0.0` to `split: 0.5`. If you want to investigate the train-test splits, this can be done in 
[plots/visualize_spectra_norman_1.ipynb](notebooks/plots/visualize_spectra_norman_1.ipynb)

### Evaluation

If you want to evaluate the test set with specific model weights, run the following 
command with the path to the checkpoint file

```bash
python src/eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

## Evaluating on differentially expressed gene (DEGs) 

**WIP**: we are working on integrating this workflow into the main pipeline. Meanwhile, 
you can follow the steps below to evaluate a perturbation on DEGs.

Step 1) Calculate significant perturbations with
E-test [notebooks/preprocessing/significant_perts_edist.ipynb](notebooks/preprocessing/significant_perts_edist.ipynb)

Step 2) Calculate differentially expressed genes for all significant
perturbations [notebooks/preprocessing/diff_exp_refactored.ipynb](notebooks/preprocessing/diff_exp_refactored.ipynb)

Step 3) Prepare the inference config [configs/experiment/mlp_norman_inference.yaml](configs/experiment/mlp_norman_inference.yaml) with the following parameters:

- Add the path to the .ckpt file
- Add the model you want to use
- Add the perturbation to be inspected
- Set the proper split and replicate corresponding to the perturbation. 
- Update the corresponding hidden and embedding dimensions

Step 4) Run `eval.py` with the inference config file.

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


