# Use an official PyTorch image with CUDA 12.1 and Python 3.11
# Note: Previous tags were not found, using 2.3.1 instead.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
EXPOSE 22

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by requirements.txt (git for transformers, mediainfo)
# Also install build-essential for potential C extensions during pip install
# Install libgl1-mesa-glx and libglib2.0-0 for OpenCV dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends git mediainfo build-essential libgl1-mesa-glx libglib2.0-0 openssh-server sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Tailscale using the official script
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y ca-certificates curl gnupg && \
    curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/focal.gpg | gpg --dearmor -o /usr/share/keyrings/tailscale-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/tailscale-archive-keyring.gpg] https://pkgs.tailscale.com/stable/ubuntu focal main" | tee /etc/apt/sources.list.d/tailscale.list && \
    apt-get update && \
    apt-get install -y tailscale && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# RUN mkdir /var/run/sshd
# COPY --chown=root:root --chmod=700 10-sshd /etc/sudoers.d/10-ssh
# RUN groupadd -r zeruser; useradd -r -g zeruser -d /app zeruser -s /bin/bash; chown -R zeruser:zeruser /app; cd /app
# delete ssh host keys so they can be generated at runtime
# RUN rm -v /etc/ssh/ssh_host_*

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This assumes your Dockerfile is in the root of your project
# and includes run_embedding_image_worker.py, workflows/, lib/, etc.
COPY . .

# Create entrypoint script to handle Tailscale setup and application startup
# COPY --chown=root:root --chmod=777 entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
# USER zeruser

COPY authorized_keys /tmp/authorized_keys || true
RUN if [ -f /tmp/authorized_keys ]; then \
    mkdir -p /root/.ssh && \
    mv /tmp/authorized_keys /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh && \
    chmod 600 /root/.ssh/authorized_keys; \
    fi

# Use the entrypoint script to handle Tailscale and application startup
ENTRYPOINT ["/app/entrypoint.sh"]
