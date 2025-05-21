# Quartet: Native FP4 Training Can Be Optimal for Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2505.14669)

## Quickstart 

Create a conda environment and install dependencies (we recommend Python 3.11):

```bash
conda create -n env python=3.11
conda activate env
```

Install the requirements (we recommend to install torch from specific channels and compile `fast_hadamard_transform` from source):

```bash
pip install -r requirements.txt
```

Run a pseudo-quantization e2e MXFP4 pre-training with:
```bash
bash main_setup.sh
```

The above command trains a 30M parameters model with the Llama-style architecture on 3B tokens.


## MXFP4 Kernels

Coming soon...
