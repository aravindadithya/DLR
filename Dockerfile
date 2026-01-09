FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git git-lfs curl iproute2 tini build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work

RUN count=0; while [ $count -lt 5 ]; do \
    pip3 install --no-cache-dir \
    torch==2.6.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu124 && break; \
    count=$((count+1)); echo "PyTorch install failed, retrying ($count/5)..."; \
    sleep 5; \
done


COPY requirements.txt .
RUN count=0; while [ $count -lt 5 ]; do \
    # Install requirements + Jupyter tools for Terminal support
    pip3 install --no-cache-dir -r requirements.txt jupyterlab visdom terminado && \
    # CLEANUP: Remove build-tools and caches in the same layer they are finished being used
    apt-get purge -y build-essential python3-dev && \
    apt-get autoremove -y && \
    rm -rf /root/.cache/pip && \
    break; \
    count=$((count+1)); echo "Requirements install failed, retrying ($count/5)..."; \
    sleep 5; \
done

# --- FINAL CONFIGURATION ---
COPY start.sh /work/start.sh
RUN chmod +x /work/start.sh

EXPOSE 8097 8888

ENTRYPOINT ["/usr/bin/tini", "--", "/bin/bash", "/work/start.sh"]
CMD ["learn_torch", "master", "0"]