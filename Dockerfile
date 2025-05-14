# ---------------------------------------------------------------------------
#   Use an official PyTorch image with CUDA 12.1 and Python 3.11
#   Note: Previous tags were not found, using 2.3.1 instead.
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
EXPOSE 22

# ---- base system packages --------------------------------------------------
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    wget git mediainfo build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    openssh-server sudo \
    python3 python3-pip python-is-python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- CUDA Toolkit ----------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg ca-certificates && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin -O /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Tailscale + SSH extras -------------------------------------------------
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/focal.gpg | \
    gpg --dearmor -o /usr/share/keyrings/tailscale-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/tailscale-archive-keyring.gpg] \
    https://pkgs.tailscale.com/stable/ubuntu focal main" \
    > /etc/apt/sources.list.d/tailscale.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends tailscale && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# rest of stack (no --pre, ordinary indexes)
COPY requirements.txt /tmp/requirements.txt
RUN DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade packaging && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# ---- application files -----------------------------------------------------
WORKDIR /app
COPY ./bin /app/bin
COPY ./lib /app/lib
COPY ./models /app/models
COPY ./prompts /app/prompts
COPY ./workflows /app/workflows
COPY ./__main__.py /app/__main__.py
COPY ./scripts/entrypoint.sh /app/entrypoint.sh

# COPY authorized_keys /root/.ssh/authorized_keys
# RUN chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# ---- launch ---------------------------------------------------------------
ENTRYPOINT ["/app/entrypoint.sh"]
