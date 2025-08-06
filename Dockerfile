# Dockerfile for a multi-stage build to create a lightweight environment
# for a Python application with PyTorch, JupyterLab, and Visdom.
# The first stage handles all the heavy lifting of building and installing,
# while the second stage creates a minimal image for production.

# --- Stage 1: Builder ---
# Use the official NVIDIA CUDA base image with Ubuntu 22.04 as our build environment.
# We name this stage "builder" so we can reference it later in the multi-stage build.
FROM nvidia/cuda:12.5.0-base-ubuntu22.04 AS builder

# Update the package lists and install necessary build tools and dependencies.
# We install Python 3, pip, net-tools for network utilities, curl for downloading files,
# build-essential for compiling native extensions, and git and git-lfs for version control.
# The `rm -rf` command cleans up the apt cache to reduce the image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip net-tools curl build-essential git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file into the container's current directory.
COPY requirements.txt .

# Install PyTorch and related packages. We use a while loop to retry the installation
# up to 5 times in case of network failures. This ensures the stage is not lost
# if there's a temporary issue.
RUN count=0; while [ $count -lt 5 ]; do \
    pip3 install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && break; \
    count=$((count+1)); echo "Installation failed, retrying ($count/5)..."; \
done

# Install the dependencies from requirements.txt. A similar retry loop is used here
# to handle any failures with these packages.
RUN count=0; while [ $count -lt 5 ]; do \
    pip3 install --no-cache-dir -r requirements.txt && break; \
    count=$((count+1)); echo "Installation failed, retrying ($count/5)..."; \
done

# Install jupyterlab and visdom in their own separate RUN command, with a retry loop.
# This makes the build even more resilient to network failures.
RUN count=0; while [ $count -lt 5 ]; do \
    pip3 install --no-cache-dir jupyterlab visdom && break; \
    count=$((count+1)); echo "Installation failed, retrying ($count/5)..."; \
done

# Install Node.js and build JupyterLab assets.
# Node.js is required for building the JupyterLab front-end.
# The `jupyterlab build` command compiles the necessary static assets.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    python3 -m jupyterlab build --dev-build=False --minimize=True

# --- Stage 2: Final Image ---
# Start a new, fresh image from the same base, which will be our final, smaller image.
# This image will not contain the build tools, reducing its final size.
FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# Update the package lists and install only the necessary runtime dependencies.
# We need python3, net-tools, git, git-lfs, and python3-tk for Tkinter support.
# Again, we clean up the apt cache to keep the image minimal.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 net-tools git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages from the "builder" stage to the final image.
# This is the key part of the multi-stage build, transferring only the necessary
# artifacts (the installed packages) without the build tools and intermediate files.
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Create a directory for JupyterLab assets and copy them from the "builder" stage.
RUN mkdir -p /usr/share/jupyter/lab
COPY --from=builder /usr/local/share/jupyter/lab /usr/share/jupyter/lab

# Copy the startup script into the container's PATH and make it executable.
COPY start.sh /work/start.sh
RUN chmod +x /work/start.sh

# Set the default working directory for the container.
WORKDIR /work

# Clone the repository with a higher timeout and skip LFS files to prevent hanging.
# We set a custom timeout of 600 seconds (10 minutes) for the git operation.
# The GIT_LFS_SKIP_SMUDGE=1 environment variable prevents LFS files from being downloaded.
#RUN git config --global http.lowSpeedLimit 0 && \
    #git config --global http.lowSpeedTime 600 && \
    #GIT_LFS_SKIP_SMUDGE=1 git clone --branch research4 https://github.com/aravindadithya/DLR .

# Expose the ports for JupyterLab (8888) and Visdom (8097).
# This tells Docker which ports the container listens on at runtime.
EXPOSE 8888 8097

# Set an environment variable for the Python path.
ENV PYTHONPATH="/work/DLR"

# Define the command that will be run when the container starts.
# It will execute the `start.sh` script we copied earlier.
CMD ["/work/start.sh"]
