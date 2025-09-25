FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential software-properties-common curl wget git unzip nano vim \
    python3 python3-pip python3-venv python3-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements-video.txt .

# Install dependencies from your requirements file
RUN pip install --upgrade pip && \
    pip install Cython "numpy<1.22" && \
    pip install -r requirements-video.txt && \
    pip install jupyter

COPY README.md .
# For testing purposes only -> The sample jupyter notebook uses
# this directory, you may remove it if you choose a superset of
# the given subset of images.
COPY MOT17/train/MOT17-04-DPM/ /MOT17/train/MOT17-04-DPM/
COPY sample.ipynb .

# You can choose to reduce the size of the dataset by copying only a subset of the images
# Add the MOT17 dataset to the directory as directed in the ReadMe file.
COPY MOT17/ /MOT17/

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]