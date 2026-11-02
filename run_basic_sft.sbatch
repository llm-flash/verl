#!/bin/bash
#SBATCH --job-name=verl_basic_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gpus=2
#SBATCH --time=01:00:00
#SBATCH --qos=standard
#SBATCH --output=logs/basic_sft_%j.log

# Minimal SFT smoke test: Qwen2.5-1.5B-Instruct on GSM8K (messages format), FSDP engine via torchrun.
set -xeuo pipefail

module load CUDA/13.0.0
cd /scratch/user/debajoym98_tamu.edu/verl

DATA_DIR=${DATA_DIR:-/scratch/user/debajoym98_tamu.edu/verl_data/gsm8k_sft}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
NGPUS=${NGPUS:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_DIR=${SAVE_DIR:-/scratch/user/debajoym98_tamu.edu/verl/checkpoints/verl_basic_sft/qwen2_5_1_5b_gsm8k_sft}

uv run --frozen --all-packages --extra vllm --extra fsdp torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} \
    -m verl.trainer.sft_trainer \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.messages_key=messages \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=1024 \
    optim.lr=1e-5 \
    engine=fsdp \
    model.path=${MODEL_PATH} \
    model.use_remove_padding=true \
    model.enable_gradient_checkpointing=true \
    checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.logger='["console"]' \
    trainer.project_name=verl_basic_sft \
    trainer.experiment_name=qwen2_5_1_5b_gsm8k_sft \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"
