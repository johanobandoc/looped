#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100l:1     ###SBATCH --gres=gpu:rtx8000:1 --gres=gpu:a100l:1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL

# Load required modules
cd /scratch/j/johan.ceron/icml2026/nanochat

module load cudatoolkit/11.6
source .venv/bin/activate
uv sync --extra gpu
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
export HF_HOME=/network/scratch/j/johan.ceron/hf
export HF_DATASETS_CACHE=/network/scratch/j/johan.ceron/hf_datasets
export TRANSFORMERS_CACHE=/network/scratch/j/johan.ceron/hf_transformers
export TORCH_HOME=/network/scratch/j/johan.ceron/torch
export WANDB_DIR=/network/scratch/j/johan.ceron/wandb
mkdir -p \
  /network/scratch/j/johan.ceron/hf \
  /network/scratch/j/johan.ceron/hf_datasets \
  /network/scratch/j/johan.ceron/hf_transformers \
  /network/scratch/j/johan.ceron/torch \
  /network/scratch/j/johan.ceron/wandb

#bash speedrun.sh

export OMP_NUM_THREADS=1
#export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_BASE_DIR=/network/scratch/j/johan.ceron/nanochat_cache

mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#python -m nanochat.report reset

if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
NPROC_PER_NODE=1
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=16 --run=$WANDB_RUN

# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ -f "$IDENTITY_FILE" ]; then
    echo "identity_conversations.jsonl already exists, skipping download..."
elif ! curl -fL -o "$IDENTITY_FILE" https://raw.githubusercontent.com/TrelisResearch/nanochat/master/identity_conversations.jsonl; then
    echo "Download failed, generating identity_conversations.jsonl locally... Make sure you have added an OpenRouter api key to openroutertoken.txt"
    PYTHONPATH=$(pwd) python dev/gen_synthetic_data.py
fi

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
python -m nanochat.report generate
