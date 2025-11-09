# --- Stage 1: Builder ---
# Use the official NVIDIA CUDA base image with Ubuntu 22.04 as our build environment.
FROM nvidia/cuda:12.5.0-base-ubuntu22.04 AS builder

# Update the package lists and install necessary build tools and dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip net-tools curl build-essential git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file into the container's current directory.
COPY requirements.txt .

# Install PyTorch and related packages. We use a while loop to retry the installation
RUN count=0; while [ $count -lt 5 ]; do \
    pip3 install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && break; \
    count=$((count+1)); echo "Installation failed, retrying ($count/5)..."; \
done

# Install the dependencies from requirements.txt. A similar retry loop is used here
RUN count=0; while [ $count -lt 5 ]; do \
    pip3 install --no-cache-dir -r requirements.txt && break; \
    count=$((count+1)); echo "Installation failed, retrying ($count/5)..."; \
done

# Install jupyterlab and visdom in their own separate RUN command, with a retry loop.
RUN count=0; while [ $count -lt 5 ]; do \
    pip3 install --no-cache-dir jupyterlab visdom && break; \
    count=$((count+1)); echo "Installation failed, retrying ($count/5)..."; \
done

# Install Node.js and build JupyterLab assets.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    python3 -m jupyterlab build --dev-build=False --minimize=True

# --- Stage 2: Final Image ---
FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# Update the package lists and install only the necessary runtime dependencies.
# Update package lists and install necessary runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    python3 \
    net-tools \
    git \
    git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages from the "builder" stage to the final image.
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Create a directory for JupyterLab assets and copy them from the "builder" stage.
RUN mkdir -p /usr/share/jupyter/lab
COPY --from=builder /usr/local/share/jupyter/lab /usr/share/jupyter/lab

# Copy the startup script into the container's PATH and make it executable.
COPY start.sh /work/start.sh
RUN chmod +x /work/start.sh

# Set the default working directory for the container.
WORKDIR /work

# Expose the ports for JupyterLab (8888) and Visdom (8097).
# This tells Docker which ports the container listens on at runtime.
EXPOSE 8888 8097

# Set an environment variable for the Python path.
ENV PYTHONPATH="/work/DLR"

# Define the command that will be run when the container starts.
# It will execute the `start.sh` script we copied earlier.
CMD ["/work/start.sh"]
