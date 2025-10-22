# Quartet: Native FP4 Training Can Be Optimal for Large Language Models

This is the official code for the Quartet FP4 training paper [![arXiv](https://img.shields.io/badge/arXiv-2505.14669-b31b1b.svg)](https://arxiv.org/abs/2505.14669) 

**[UPDATE 28.09.25]:** Quartet has been accepted to NeurIPS 2025!

**[UPDATE 28.09.25]:** Check out our [latest work on MXFP4/NVFP4 for PTQ](https://arxiv.org/abs/2509.23202).

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

Quartet kernels are released as part of the [QuTLASS](https://github.com/IST-DASLab/qutlass) library and the [FP-Quant](https://github.com/IST-DASLab/FP-Quant) training/inference addon to [`transformers`](https://huggingface.co/docs/transformers/main/en/quantization/fp_quant) and [`vLLM`](https://github.com/vllm-project/vllm/pull/24440).

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
