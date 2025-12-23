FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update and install essential development tools
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    vim \
    nano \
    gdb \
    python3 \
    python3-pip \
    python3-venv \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages commonly used for plotting and data analysis
# Using --break-system-packages is safe in a Docker container
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    matplotlib \
    pandas \
    scipy \ 
    seaborn

# Set up a working directory
WORKDIR /workspace

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Keep container running and ready for development
CMD ["/bin/bash"]

