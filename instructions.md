# verl on the Vision cluster (TAMU)

verl setup for the Vision cluster. It uses vLLM for rollout and FSDP for training, on CUDA 13.0 and PyTorch 2.13.

## 1. Installation

Prerequisites: [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and the Vision `CUDA/13.0.0` module.

```bash
cd /scratch/user/debajoym98_tamu.edu
git clone git@github.com:llm-flash/verl.git
cd verl

module load CUDA/13.0.0
export UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
uv sync --frozen --extra vllm --extra fsdp   # creates .venv from uv.lock (~3 min)
```

This installs the following into `verl/.venv`:

| Package | Version |
|---|---|
| torch / torchvision | 2.13.0+cu130 / 0.28.0+cu130 |
| vllm | 0.32.0 |
| flash-attn | 2.8.3 |
| transformers | 5.16.1 |
| qwen-vl-utils | 0.0.14 (vision-language models) |

Choose one inference engine (`vllm` or `sglang`) and one trainer (`fsdp`, `megatron`, or `veomni`). Use the same extras for every command in a job. If they differ, `uv run` re-syncs `.venv` and reinstalls torch.

### Verify the install

```bash
module load CUDA/13.0.0
.venv/bin/python -c "import torch, vllm, flash_attn, verl; print(torch.__version__, vllm.__version__, flash_attn.__version__)"
```

Login nodes (`slogin-*`) have no GPU, so `torch.cuda.is_available()` returns `False` there. Run training through SLURM.

### Activate the environment interactively (optional)

```bash
module load CUDA/13.0.0
cd /scratch/user/debajoym98_tamu.edu/verl
source .venv/bin/activate
```

## 2. Prepare data

Run this on the login node, which has internet access:

```bash
# RL (GRPO / PPO): prompt-only data with a rule-based reward
.venv/bin/python examples/data_preprocess/gsm8k.py \
    --local_save_dir /scratch/user/debajoym98_tamu.edu/verl_data/gsm8k

# SFT: chat-format data with a "messages" column (user question + reference solution)
.venv/bin/python examples/data_preprocess/gsm8k_multiturn_sft.py \
    --local_save_dir /scratch/user/debajoym98_tamu.edu/verl_data/gsm8k_sft
```

Both write `train.parquet` (7,473 rows) and `test.parquet` (1,319 rows). The RL files store the prompt plus the ground-truth answer used by the reward function. The SFT files store a `messages` list with a `user` turn (the question plus "Let's think step by step and output the final answer after "####".") and an `assistant` turn (the full GSM8K solution ending in `#### <answer>`). SFT trains only on the assistant turn.

Other preprocessors are in `examples/data_preprocess/`. For example, `geo3k.py` prepares the dataset used by the vision-language examples.

## 3. Run a basic RL (GRPO) experiment

`run_basic_grpo.sbatch` is a small smoke test with these settings:

- Model: Qwen2.5-1.5B-Instruct
- Data: GSM8K
- Rollout: vLLM
- Training: FSDP
- Resources: 2 H200 GPUs
- Length: 10 steps
- Logging: console only

```bash
cd /scratch/user/debajoym98_tamu.edu/verl
mkdir -p logs
sbatch run_basic_grpo.sbatch

# overrides via env vars or extra Hydra args:
TOTAL_STEPS=20 MODEL_PATH=Qwen/Qwen3-4B sbatch run_basic_grpo.sbatch
sbatch run_basic_grpo.sbatch actor_rollout_ref.rollout.n=8
```

Monitor the job:

```bash
squeue -u $USER
tail -f logs/basic_grpo_<jobid>.log      # look for "step:N" lines with critic/score/mean, etc.
scancel <jobid>
```

Each launch follows the same pattern. The driver runs under `uv run`, and Ray workers use the same environment through `ray_kwargs.ray_init.runtime_env.py_executable`:

```bash
uv run --frozen --all-packages --extra vllm --extra fsdp python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=... data.val_files=... \
    actor_rollout_ref.model.path=... \
    trainer.n_gpus_per_node=<N> trainer.nnodes=1 \
    ray_kwargs.ray_init.runtime_env.py_executable="uv run --frozen --all-packages --extra vllm --extra fsdp"
```

## 4. Run a basic SFT experiment

`run_basic_sft.sbatch` fine-tunes on the GSM8K SFT data from section 2:

- Model: Qwen2.5-1.5B-Instruct
- Data: `verl_data/gsm8k_sft` (`messages` column)
- Training: FSDP engine, launched with `torchrun` (SFT does not use Ray or vLLM)
- Resources: 2 H200 GPUs
- Length: 1 epoch (about 116 steps at a global batch size of 64)
- Checkpoint: saved at the last step, including a Hugging Face copy of the weights

```bash
cd /scratch/user/debajoym98_tamu.edu/verl
mkdir -p logs
sbatch run_basic_sft.sbatch

# overrides via env vars or extra Hydra args:
TOTAL_EPOCHS=2 MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct sbatch run_basic_sft.sbatch
sbatch run_basic_sft.sbatch optim.lr=1e-4 model.lora_rank=32 model.lora_alpha=16 model.target_modules=all-linear   # LoRA
```

Monitor with `tail -f logs/basic_sft_<jobid>.log`, and watch `train/loss` and `val/loss`.

The checkpoint is written to `checkpoints/verl_basic_sft/qwen2_5_1_5b_gsm8k_sft/global_step_<N>/`, which you can change with `SAVE_DIR`. Its `huggingface/` subdirectory holds a standard Hugging Face model (config, tokenizer, and safetensors weights) that vLLM, transformers, and the PPO script can load directly.

The launch pattern is:

```bash
uv run --frozen --all-packages --extra vllm --extra fsdp torchrun --standalone --nnodes=1 --nproc_per_node=<N> \
    -m verl.trainer.sft_trainer \
    data.train_files=... data.val_files=... data.messages_key=messages \
    model.path=... engine=fsdp \
    checkpoint.save_contents='["model","optimizer","extra","hf_model"]'
```

## 5. Run a basic PPO experiment

`run_basic_ppo.sbatch` runs classic PPO with an actor and a learned critic, using GAE advantages. It uses the same RL data as GRPO (`verl_data/gsm8k`). It differs from the GRPO script in these ways:

- `algorithm.adv_estimator=gae` replaces `grpo`, and a critic model is trained alongside the actor (`critic.*` settings).
- `actor_rollout_ref.rollout.n=1`, since PPO does not need a group of samples per prompt.
- `actor_rollout_ref.actor.use_kl_loss=False`, so no KL loss term is added to the actor.
- vLLM `gpu_memory_utilization` is lowered to 0.4 to leave room for the critic.

```bash
sbatch run_basic_ppo.sbatch
TOTAL_STEPS=20 sbatch run_basic_ppo.sbatch
```

Look for `critic/vf_loss`, `critic/vf_explained_var`, and `critic/score/mean` in the `step:N` lines of `logs/basic_ppo_<jobid>.log`.

## 6. SFT, then PPO

To run RL on top of the SFT model, point PPO at the SFT checkpoint's Hugging Face directory. Use it for both the actor and the critic. The critic loads the same weights with a new value head.

```bash
sbatch run_basic_sft.sbatch
# after it finishes, use the last global_step_<N>:
SFT_CKPT=/scratch/user/debajoym98_tamu.edu/verl/checkpoints/verl_basic_sft/qwen2_5_1_5b_gsm8k_sft/global_step_116/huggingface
MODEL_PATH=$SFT_CKPT sbatch run_basic_ppo.sbatch          # PPO
MODEL_PATH=$SFT_CKPT sbatch run_basic_grpo.sbatch         # or GRPO
```

To chain the jobs automatically, use `sbatch --dependency=afterok:<sft_jobid> ...`.

If a checkpoint was saved without `hf_model`, convert the FSDP shards with the model merger:

```bash
.venv/bin/python -m verl.model_merger merge --backend fsdp \
    --local_dir checkpoints/verl_basic_sft/qwen2_5_1_5b_gsm8k_sft/global_step_<N> \
    --target_dir checkpoints/verl_basic_sft/qwen2_5_1_5b_gsm8k_sft/global_step_<N>/hf_merged
```

## 7. Larger / official examples

The scripts in `examples/` need an 8-GPU node, for example `#SBATCH --gpus=8 --cpus-per-task=48 --mem=0`. Load `CUDA/13.0.0` first, then run the scripts from the repo root:

```bash
# text LLM, GSM8K
TRAIN_FILE=/scratch/user/debajoym98_tamu.edu/verl_data/gsm8k/train.parquet \
TEST_FILE=/scratch/user/debajoym98_tamu.edu/verl_data/gsm8k/test.parquet \
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh

# vision-language model (Geo3K)
bash examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh
```

Most scripts log to wandb. Run `wandb login` first, or override with `trainer.logger='["console"]'`.

## Troubleshooting

- **`Can't initialize NVML` / no CUDA**: you are on a login node. Submit through SLURM.
- **`DataLoader worker ... is killed by signal: Killed` after the last step**: dataloader workers left running at the end of training are killed during Ray shutdown. Training results are unaffected. The basic scripts set `data.dataloader_num_workers=0` to avoid it.
- **torch reinstalls on every run**: commands in the same job used different `--extra` sets.
- **Start over**: `rm -rf .venv && uv sync --frozen --extra vllm --extra fsdp`.
- **Sandbox fallback**: `VERL_USE_UV=0` runs the example scripts with the active Python instead of `uv run`.
