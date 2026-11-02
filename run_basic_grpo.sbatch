#!/bin/bash
#SBATCH --job-name=verl_basic_grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gpus=2
#SBATCH --time=01:00:00
#SBATCH --qos=standard
#SBATCH --output=logs/basic_grpo_%j.log

# Minimal GRPO smoke test: Qwen2.5-1.5B-Instruct on GSM8K, vLLM rollout + FSDP training.
set -xeuo pipefail

module load CUDA/13.0.0
cd /scratch/user/debajoym98_tamu.edu/verl

DATA_DIR=${DATA_DIR:-/scratch/user/debajoym98_tamu.edu/verl_data/gsm8k}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
NGPUS=${NGPUS:-2}
TOTAL_STEPS=${TOTAL_STEPS:-10}

ray stop --force >/dev/null 2>&1 || true

uv run --frozen --all-packages --extra vllm --extra fsdp python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=verl_basic \
    trainer.experiment_name=qwen2_5_1_5b_gsm8k_grpo \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.total_epochs=1 \
    ray_kwargs.ray_init.runtime_env.py_executable="uv run --frozen --all-packages --extra vllm --extra fsdp" \
    "$@"
