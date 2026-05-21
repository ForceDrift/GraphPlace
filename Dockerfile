# Use an official PyTorch runtime with CUDA as a parent image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Set shell to bash for better string handling
SHELL ["/bin/bash", "-c"]

# Install system dependencies required for building C++ components (like RePlAce)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libgoogle-perftools-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
# We exclude 'macro-place' because we will install it from the provided externals subfolder
RUN grep -v "macro-place" requirements.txt > temp_reqs.txt && \
    pip install --no-cache-dir -r temp_reqs.txt && \
    rm temp_reqs.txt

# Copy the entire project into the container
COPY . .

# Install the competition evaluation harness in editable mode
RUN pip install -e ./externals/macro-place-challenge-2026

# Set environment variables
# PYTHONPATH ensures that both the core 'graphplace' modules and the 'macro_place' harness are discoverable
ENV PYTHONPATH="/app:/app/externals/macro-place-challenge-2026"

# Verify the installation and dependencies
RUN python -c "import torch; import torch_geometric; print('GraphPlace Environment: Success')"

# Expose the competition evaluation tool as the default command
# Judges can override this to run specific training or inference scripts
CMD ["python", "-m", "macro_place.evaluate", "--help"]
