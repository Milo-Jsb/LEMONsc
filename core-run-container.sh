#!/bin/bash

# Function to search available port
find_available_port() {
    local start_port=${1:-8889}
    local port=$start_port
    
    # Use 'ss' instead of 'netstat' (more commonly available)
    while ss -tuln | grep -q ":$port "; do
        echo "Port $port unavailable, trying the next one..." >&2
        ((port++))
    done
    
    echo "Port found: $port" >&2
    echo $port
}

# Find available port for container exposure
AVAILABLE_PORT=$(find_available_port 8889)
echo "Available port found: $AVAILABLE_PORT"

# Ask user for confirmation with custom option
echo "Options:"
echo "1. Use suggested port: $AVAILABLE_PORT"
echo "2. Enter custom port"
echo "3. Cancel"
read -p "Choose option (1/2/3): " -n 1 -r
echo

case $REPLY in
    1)
        echo "Using suggested port: $AVAILABLE_PORT"
        ;;
    2)
        read -p "Enter custom port number: " CUSTOM_PORT
        # Validate custom port
        if [[ ! $CUSTOM_PORT =~ ^[0-9]+$ ]] || [ $CUSTOM_PORT -lt 1 ] || [ $CUSTOM_PORT -gt 65535 ]; then
            echo "Invalid port number. Must be between 1-65535"
            exit 1
        fi
        # Check if custom port is available
        if ss -tuln | grep -q ":$CUSTOM_PORT "; then
            echo "Port $CUSTOM_PORT is already in use!"
            exit 1
        fi
        AVAILABLE_PORT=$CUSTOM_PORT
        echo "Using custom port: $AVAILABLE_PORT"
        ;;
    3)
        echo "Container startup cancelled."
        exit 1
        ;;
    *)
        echo "Invalid option. Cancelling."
        exit 1
        ;;
esac

echo "Starting container with port $AVAILABLE_PORT..."

# Run container
docker run \
    --rm \
    -it \
    --name lemonsc \
    --gpus all \
    --ipc=host \
    --user appuser \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace \
    -p $AVAILABLE_PORT:8889 \
    core \