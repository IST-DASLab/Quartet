#!/bin/bash
set -e

# common env
export VOCAB_SIZE=32000
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512
export DATASET="c4"

# # 30M
# export N_LAYER=6
# export N_EMBD=640
# export N_HEAD=5
# export LR=0.0012
# export BASE_TOKENS=3000000000 # 3B
# export MODEL_SIZE_PREFIX="30M"

# # 50M
# export N_LAYER=7
# export N_EMBD=768
# export N_HEAD=6
# export LR=0.0012
# export BASE_TOKENS=5000000000 # 5B
# export MODEL_SIZE_PREFIX="50M"

# 100M
export N_LAYER=8
export N_EMBD=1024
export N_HEAD=8
export LR=0.0006
export BASE_TOKENS=10000000000 # 10B
export MODEL_SIZE_PREFIX="100M"

# # 200M
# export N_LAYER=10
# export N_EMBD=1280
# export N_HEAD=10
# export LR=0.0003
# export BASE_TOKENS=20000000000 # 20B
# export MODEL_SIZE_PREFIX="200M"

# quant and backward‐scheme (unchanged)
export W_QUANT="QuestMXFP4Quantizer"
export W_BITS=4
export W_QUANT_KWARGS="{}"
export A_QUANT="QuestMXFP4Quantizer"
export A_BITS=4
export A_QUANT_KWARGS="{}"
export G_QUANT="AlbertTsengQuantizer"
export G_BITS=4
export HADAMARD_DIM=32
export G_QUANT_KWARGS="{\"hadamard_dim\": ${HADAMARD_DIM}, \"rerotate\": \"signs\", \"stochastic\": true}"
export BACKWARD_SCHEME="Q(E)Q(Wt)t_Q(Et)Q(Xt)t"
# export BACKWARD_SCHEME="EW_EtX"
export BACKWARD_SCHEME_KWARGS="{}"

# build a list: ¼×, ½×, 1×, 2×, 4×, 8×
TOKENS_LIST=(
  $((BASE_TOKENS/4))
  $((BASE_TOKENS/2))
  ${BASE_TOKENS}
  $((BASE_TOKENS*2))
  $((BASE_TOKENS*4))
  $((BASE_TOKENS*8))
)

# main loop
for TOKENS in "${TOKENS_LIST[@]}"; do
  export TOKENS
  export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
  export WARMUP_STEPS=$((ITERATIONS / 10))

  # include TOKENS in your wandb run‐name so you can distinguish runs
  WANDB_PREFIX="UNTIED-${MODEL_SIZE_PREFIX}-TOK${TOKENS}-${W_QUANT}@${W_BITS}:${A_QUANT}@${A_BITS}:${G_QUANT}@${G_BITS}:${BACKWARD_SCHEME}-${DATASET}"

  echo "===== running with TOKENS=${TOKENS} (iters=${ITERATIONS}, warmup=${WARMUP_STEPS}) ====="

  TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --datasets-dir /dev/shm \
    --latest-ckpt-interval 1000 \
    --model llama \
    --compile \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --wandb \
    --wandb-project "backward-schemes" \
    --wandb-run-prefix "${WANDB_PREFIX}" \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --warmup-steps ${WARMUP_STEPS} \
    --iterations ${ITERATIONS} \
    --lr ${LR} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --a-quant-kwargs "${A_QUANT_KWARGS}" \
    --g-quant ${G_QUANT} \
    --g-quant-kwargs "${G_QUANT_KWARGS}" \
    --backward-scheme ${BACKWARD_SCHEME} \
    --backward-scheme-kwargs "${BACKWARD_SCHEME_KWARGS}"
done
