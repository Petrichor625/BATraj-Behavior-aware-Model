# BAT: Behavior-Aware Human-Like Trajectory Prediction for Autonomous Driving

Official implementation of **BAT: Behavior-Aware Human-Like Trajectory Prediction for Autonomous Driving**
Accepted at **AAAI 2024**

------

## Overview

This repository contains the official implementation of **BAT**, a behavior-aware trajectory prediction framework for autonomous driving.

> **Paper**: *BAT: Behavior-Aware Human-Like Trajectory Prediction for Autonomous Driving*
> **Venue**: AAAI 2024

------

## Important Notice

> [!WARNING]
> Due to the loss of an earlier internal code version, this repository is a carefully reconstructed implementation of the original project. The current version has been rebuilt as faithfully as possible based on our available codebase and experimental pipeline, with reconstruction grounded on **STDAN**. In our reproduced experiments, the overall model performance is slightly better than the results reported in the original paper.

We sincerely apologize for any inconvenience this may cause.

If you would like to quickly reproduce stronger results, please also refer to our latest work: **HLTP**.

👉 **[HLTP Project on GitHub](https://github.com/Petrichor625/HLTP)** 👈

------

## Installation

### Environment

The codebase has been tested on the following setup:

- **Ubuntu 20.04**
- **CUDA 11.7**
- **Python 3.8**
- **PyTorch**

### 1. Clone the repository

```bash
git clone https://github.com/Petrichor625/BATraj-Behavior-aware-Model.git
cd BATraj-Behavior-aware-Model
```

### 2. Install dependencies

If you already have a Python environment prepared, install the required packages with:

```bash
pip install -r requirements.txt
```

### 3. Create the Conda environment

We also provide a Conda environment file in the `environments/` directory:

```bash
cd environments
conda env create -f environment.yml
```

If the above command does not work properly on your system, you may manually create the environment and install the main dependencies:

```bash
conda create -n behavior_aware python=3.8
conda activate behavior_aware
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

------



## Configuration

Before training or evaluation, please update the relevant paths in the source code.

### 1. Checkpoint path

In `config_new.py`, set:

```python
args['path']
```

to the directory where model checkpoints will be stored.

### 2. Dataset path

Please update the dataset path accordingly, for example:

```python
lo.NgsimDataset('../NGSIM/TrainSet.mat')
```

Make sure the path matches the actual location of the dataset on your machine.

------

## Training

If you want to resume training from pretrained checkpoints, specify the checkpoint paths in `train5f_behavior_new.py`:

```python
generator.load_state_dict(t.load('../epoch8_g.tar'))
gdEncoder.load_state_dict(t.load('../epoch8_gd.tar'))
```

If you want to train the model from scratch, simply comment out these lines.

------

## Evaluation

To evaluate a trained model, edit `evaluate5f_behavior_new.py` and specify the checkpoint epoch number:

```python
if __name__ == '__main__':
    names = ['XX']  # e.g. ['8']
    evaluate = Evaluate()

    for epoch in names:
        evaluate.main(name=epoch, val=False)
```

Replace `XX` with the corresponding checkpoint epoch number you want to evaluate.

------

## Reproducibility Notes

Since this repository is a reconstructed version rather than the exact original code used in the paper, minor implementation and performance differences may exist. Nevertheless, we have made every effort to preserve the original methodology and reproduce the reported behavior as closely as possible.

------

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@inproceedings{liao2024bat,
  title={Bat: Behavior-aware human-like trajectory prediction for autonomous driving},
  author={Liao, Haicheng and Li, Zhenning and Shen, Huanming and Zeng, Wenxuan and Liao, Dongping and Li, Guofa and Xu, Chengzhong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={9},
  pages={10332--10340},
  year={2024}
}
```

------

