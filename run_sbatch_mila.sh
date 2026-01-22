#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100l:1     ###SBATCH --gres=gpu:rtx8000:1 --gres=gpu:a100l:1
#SBATCH --partition=lab-bengioy
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
cd /home/mila/j/johan.ceron/scratch/icml2026/nanochat

# -----------------------------------------------------------------------------
# Modules / env
# -----------------------------------------------------------------------------
module load cudatoolkit/11.6

export HF_HOME=/network/scratch/j/johan.ceron/hf
export HF_DATASETS_CACHE=/network/scratch/j/johan.ceron/hf_datasets
export TRANSFORMERS_CACHE=/network/scratch/j/johan.ceron/hf_transformers
export TORCH_HOME=/network/scratch/j/johan.ceron/torch
export WANDB_DIR=/network/scratch/j/johan.ceron/wandb
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/network/scratch/j/johan.ceron/nanochat_cache
# -----------------------------------------------------------------------------
# uv cache (evita disk quota en $HOME)
# -----------------------------------------------------------------------------
export UV_CACHE_DIR=/network/scratch/j/johan.ceron/uv_cache
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR"


mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$WANDB_DIR" \
  "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# uv + venv
# -----------------------------------------------------------------------------
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# -----------------------------------------------------------------------------
# Weights & Biases
# -----------------------------------------------------------------------------
export WANDB_PROJECT="nanochat_speedrun_sigreg"
# export WANDB_ENTITY="johan-ceron-obando"   # opcional
# : "${WANDB_RUN:=R4SI_v2_dim_512_iter7125}"  # si no está seteado, usa este
export WANDB_ENTITY="johan-ceron-obando"   # opcional
: "${WANDB_RUN:=control}"

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------
# Install Rust / Cargo en scratch (evita disk quota en $HOME)
export CARGO_HOME=/network/scratch/j/johan.ceron/cargo
export RUSTUP_HOME=/network/scratch/j/johan.ceron/rustup
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
export PATH="$CARGO_HOME/bin:$PATH"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$CARGO_HOME/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download dataset in parallel + build tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# -----------------------------------------------------------------------------
# Train + eval
# -----------------------------------------------------------------------------
NPROC_PER_NODE=1

# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
#   --num_iterations=7125 \
#   --n_recur_block=4 \
#   --device_batch_size=32 \
#   --model_tag="R4SIv2dim512" \
#   --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
  --num_iterations=7125 \
  --n_prelude=12 \
  --n_coda=0 \
  --n_recur_block=0 \
  --model_tag="control" \
  --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# Identity conversations
# -----------------------------------------------------------------------------
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ -f "$IDENTITY_FILE" ]; then
    echo "identity_conversations.jsonl already exists, skipping download..."
elif ! curl -fL -o "$IDENTITY_FILE" https://raw.githubusercontent.com/TrelisResearch/nanochat/master/identity_conversations.jsonl; then
    echo "Download failed, generating identity_conversations.jsonl locally..."
    echo "Make sure you have added an OpenRouter api key to openroutertoken.txt"
    PYTHONPATH=$(pwd) python dev/gen_synthetic_data.py
fi

# -----------------------------------------------------------------------------
# Midtraining + SFT + report
# -----------------------------------------------------------------------------
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

python -m nanochat.report generate
