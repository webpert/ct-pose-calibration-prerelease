FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONPATH=/workspace/src:${PYTHONPATH} \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libassimp-dev \
    libboost-all-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglew-dev \
    libgomp1 \
    libgtk-3-dev \
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    tzdata \
    unzip \
    wget \
    && ln -fs /usr/bin/python3 /usr/bin/python \
    && ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 \
    torchvision==0.15.1+cu118 \
    torchaudio==2.0.1+cu118

COPY src /workspace/src

WORKDIR /workspace/src

RUN git clone https://gitlab.inria.fr/bkerbl/simple-knn.git pose_gaussian/submodules/simple-knn && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt scikit-learn==1.3.2 && \
    python3 -m pip install --no-cache-dir --no-build-isolation pose_gaussian/submodules/simple-knn && \
    python3 -m pip install --no-cache-dir --no-build-isolation pose_gaussian/submodules/self-xray-gaussian-rasterization-voxelization

RUN python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

CMD ["/bin/bash"]
