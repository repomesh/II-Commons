#!/bin/bash
set -e

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
  echo "Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
fi

# Start Tailscale if TAILSCALE_AUTHKEY is provided
if [ -n "$TAILSCALE_AUTHKEY" ]; then
  echo "Starting Tailscale..."
  tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &

  # Set default hostname if not provided
  if [ -z "$TAILSCALE_HOSTNAME" ]; then
    # Generate a hostname based on container ID and application name
    CONTAINER_ID=$(hostname)
    TAILSCALE_HOSTNAME="chipmunk-worker-${CONTAINER_ID:0:8}"
    echo "Auto-generated Tailscale hostname: $TAILSCALE_HOSTNAME"
  fi

  # Start Tailscale with appropriate flags
  tailscale up --authkey=$TAILSCALE_AUTHKEY --hostname=$TAILSCALE_HOSTNAME
  echo "Tailscale: started successfully"
else
  echo "Tailscale: `TAILSCALE_AUTHKEY` not provided, skipping setup"
fi

# Start SSH server if SSH_PUBKEY is provided
# if [[ -z "${SSH_PUBKEY}" ]]; then
# 	echo "No SSH_PUBKEY set, not starting sshd"
# else
# 	echo "Generating host keys"
# 	 /usr/bin/sudo /usr/sbin/dpkg-reconfigure openssh-server > /dev/null 2>&1
# 	echo "Starting sshd"
# 	/usr/bin/sudo /usr/sbin/sshd -D &
# fi

# # Create authorized_keys file if SSH_PUBKEY is provided
# if [[ -z "${SSH_PUBKEY}" ]]; then
# 	echo "No SSH_PUBKEY set, not creating authorized_keys"
# else
# 	mkdir -p ~/.ssh
# 	echo -e $SSH_PUBKEY > ~/.ssh/authorized_keys
# 	chmod 700 ~/.ssh
# 	chmod 600 ~/.ssh/authorized_keys
# fi

# Run the main application
echo "Chipmunk: starting application..."
exec python . "$@"
