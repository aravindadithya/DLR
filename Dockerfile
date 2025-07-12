FROM nvidia/cuda:12.5.0-base-ubuntu22.04 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip net-tools curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir --resume-retries 5 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir --resume-retries 5 -r requirements.txt && \
    pip3 install --no-cache-dir jupyterlab visdom


RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    python3 -m jupyterlab build --dev-build=False --minimize=True


FROM nvidia/cuda:12.5.0-base-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 net-tools && \
    rm -rf /var/lib/apt/lists/*


COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

RUN mkdir -p /usr/share/jupyter/lab

COPY --from=builder /usr/local/share/jupyter/lab /usr/share/jupyter/lab

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

WORKDIR /workspace

COPY . .

EXPOSE 8888 8097

ENV PYTHONPATH="/workspace"

CMD ["/usr/local/bin/start.sh"]