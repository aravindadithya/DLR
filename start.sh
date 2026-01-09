#!/bin/bash

# --- Inputs ---
REPO_NAME=$1
REPO_BRANCH=${2:-"master"}
SKIP_LFS=${3:-0}

if [ -z "$REPO_NAME" ]; then
    echo "ERROR: Repository name is required."
    exit 1
fi

echo "--- Environment Sanity Check ---"
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.is_available()}')"

# --- Repo Logic ---
REPO_DIR="/work/$REPO_NAME"
# Ensure the URL is correct. If private, use a Token.
REPO_URL="https://github.com/aravindadithya/$REPO_NAME.git"

git config --global --add safe.directory "$REPO_DIR"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Updating $REPO_NAME ($REPO_BRANCH)..."
    cd "$REPO_DIR" || exit 1
    GIT_LFS_SKIP_SMUDGE=$SKIP_LFS git pull origin "$REPO_BRANCH"
else
    echo "Cloning $REPO_NAME ($REPO_BRANCH)..."
    cd /work || exit 1
    # If this fails, the repo is likely private or the URL is wrong
    GIT_LFS_SKIP_SMUDGE=$SKIP_LFS git clone --branch "$REPO_BRANCH" "$REPO_URL" || echo "Clone failed! Check repo visibility."
fi

cd /work

# --- Start Visdom ---
nohup python3 -m visdom.server -port 8097 > /work/visdom.log 2>&1 &
echo "Visdom started on port 8097"

# --- Start Jupyter Lab ---
echo "Starting Jupyter Lab on port 8888..."
export SHELL=/bin/bash

# FIXED: Using a more robust way to pass settings to avoid shell expansion errors
exec python3 -m jupyterlab \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ip=0.0.0.0 \
    --ServerApp.terminado_settings='{"shell_command": ["/bin/bash"]}'