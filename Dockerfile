FROM python:3.11-slim-bookworm

# Install system dependencies
# ffmpeg: required for audio I/O
# libsndfile1: required by soundfile
# git: required for some pip packages
# build-essential: for compiling extensions if needed
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy project files
COPY pyproject.toml .
# Copy source code
COPY solomuse_data/ solomuse_data/

# Install the package
# We also explicitly install 'demucs' to ensure the optional feature works out of the box
RUN pip install --no-cache-dir . demucs

# Default command (can be overridden)
ENTRYPOINT ["solomuse-data"]
CMD ["--help"]
