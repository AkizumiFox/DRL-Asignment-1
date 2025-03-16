FROM ubuntu:20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set up time zone
RUN apt-get update && apt-get install -y tzdata
ENV TZ=UTC

# Install system dependencies, including HDF5 libraries needed for h5py
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install dependencies in smaller groups to isolate potential issues
RUN pip3 install numpy matplotlib pandas scipy
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install h5py
RUN pip3 install tensorflow
RUN pip3 install gym stable-baselines3
RUN pip3 install jupyter ipykernel

# Set the working directory
WORKDIR /app

# Create a non-root user
RUN useradd -m drluser
RUN chown -R drluser:drluser /app
USER drluser

# Command to run when container starts
CMD ["/bin/bash"] 