#!/bin/bash

# Script to start Visdom server in the background and Jupyter Lab in the foreground.

# Function to check if a process is already running on a port
is_port_in_use() {
  # Prefer 'ss' from iproute2 over 'netstat' from net-tools as 'ss' is more modern.
  # If 'ss' is not found, fallback to 'netstat'.
  if command -v ss &> /dev/null; then
    ss -tuln | grep ":$1 " > /dev/null
  elif command -v netstat &> /dev/null; then
    netstat -tuln | grep ":$1 " > /dev/null
  else
    echo "Neither ss nor netstat found. Cannot check port status reliably." >&2
    return 1 # Indicate failure to check
  fi
}

echo "--- Starting Services ---"

# --- Start Visdom Server ---
VISDOM_PORT=8097
echo "Attempting to start Visdom server on port $VISDOM_PORT..."

if is_port_in_use "$VISDOM_PORT"; then
  echo "WARNING: Port $VISDOM_PORT is already in use. Visdom might not start correctly."
else
  # Start Visdom in the background using the Python module directly.
  # No 'conda run' needed as we are not in a conda environment.
  python3 -m visdom.server -port "$VISDOM_PORT" &
  # Capture the PID of the background process.
  VISDOM_PID=$!
  echo "Visdom server started with PID: $VISDOM_PID"
fi

# --- Start Jupyter Lab Server ---
JUPYTER_PORT=8888
echo "Attempting to start Jupyter Lab on port $JUPYTER_PORT..."

if is_port_in_use "$JUPYTER_PORT"; then
  echo "ERROR: Port $JUPYTER_PORT is already in use. Cannot start Jupyter Lab. Exiting."
  exit 1
else
  # 'exec' replaces the current shell process with the Jupyter Lab process.
  # This is crucial for Docker containers for graceful shutdown.
  # Use python3 -m jupyterlab to call it as a module.
  echo "Jupyter Lab will be accessible via http://localhost:$JUPYTER_PORT/ (check logs for token)"
  exec python3 -m jupyterlab --port="$JUPYTER_PORT" --no-browser --allow-root --ip=0.0.0.0
fi

# Any commands placed here will NOT be executed if 'exec' is successful,
# as 'exec' replaces the current process.