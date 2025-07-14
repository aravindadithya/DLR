#!/bin/bash

REPO_URL="https://github.com/aravindadithya/DLR.git"
CLONE_DIR="DLR"
BRANCH_NAME="research4"
VISDOM_PORT="8097"

echo "Checking for Git installation..."
if ! command -v git &> /dev/null; then
    echo "Git not found. Attempting to install Git..."

    if command -v apt &> /dev/null; then
        echo "Detected apt. Installing Git using apt..."
        sudo apt update
        sudo apt install -y git
    elif command -v yum &> /dev/null; then
        echo "Detected yum. Installing Git using yum..."
        sudo yum install -y git
    elif command -v dnf &> /dev/null; then
        echo "Detected dnf. Installing Git using dnf..."
        sudo dnf install -y git
    else
        echo "Error: Neither apt, yum, nor dnf package manager found."
        echo "Please install Git manually for your operating system."
        exit 1
    fi

    if ! command -v git &> /dev/null; then
        echo "Error: Git installation failed. Please install Git manually."
        exit 1
    else
        echo "Git installed successfully."
    fi
else
    echo "Git is already installed."
fi

echo "Cloning repository '$REPO_URL' (branch: $BRANCH_NAME) without Git LFS..."

if [ -d "$CLONE_DIR" ]; then
    echo "Warning: Directory '$CLONE_DIR' already exists. Removing it."
    rm -rf "$CLONE_DIR"
fi

git clone --depth 1 --no-checkout --branch "$BRANCH_NAME" "$REPO_URL" "$CLONE_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone repository."
    exit 1
fi

cd "$CLONE_DIR" || { echo "Error: Failed to change directory to $CLONE_DIR"; exit 1; }

git config lfs.dontdownload true
git checkout "$BRANCH_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Failed to checkout branch '$BRANCH_NAME' in $CLONE_DIR."
    exit 1
fi
echo "Repository cloned successfully to $CLONE_DIR and branch '$BRANCH_NAME' checked out."

echo "Installing pip requirements..."
python -m ensurepip --upgrade > /dev/null 2>&1
python -m pip install --upgrade pip > /dev/null 2>&1

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install pip requirements."
        exit 1
    fi
    echo "Pip requirements installed successfully."
else
    echo "Warning: requirements.txt not found in $CLONE_DIR. Skipping pip installation."
fi

PARENT_WORKSPACE_DIR=$(dirname "$(pwd)")
DLR_PATH="$PARENT_WORKSPACE_DIR/$CLONE_DIR"

export PYTHONPATH="$PYTHONPATH:$DLR_PATH"
echo "PYTHONPATH set to: $PYTHONPATH"

echo "Starting Visdom server on port $VISDOM_PORT..."
nohup python -m visdom.server -port "$VISDOM_PORT" > visdom.log 2>&1 &
VISDOM_PID=$!
echo "Visdom server started with PID: $VISDOM_PID. Check visdom.log for output."
echo "Access Visdom at: http://localhost:$VISDOM_PORT"

echo "Script execution complete."