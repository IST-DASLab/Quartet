# Quartet: Native FP4 Training Can Be Optimal for Large Language Models

This is the official code for the Quartet FP4 training paper [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2505.14669) 

Currently, this repository is work in progress. We provide code for reproducing the accuracy experiments, and will release kernels for the performance experiments at a later time.

This work was presented at the GPU MODE lecture cycle [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/watch?v=XVo17Q7YapA)

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

## Cite This Work
```
@misc{castro2025quartetnativefp4training,
      title={Quartet: Native FP4 Training Can Be Optimal for Large Language Models}, 
      author={Roberto L. Castro and Andrei Panferov and Soroush Tabesh and Oliver Sieberling and Jiale Chen and Mahdi Nikdan and Saleh Ashkboos and Dan Alistarh},
      year={2025},
      eprint={2505.14669},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.14669}, 
}
```
