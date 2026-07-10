FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
ENV PYTHONPATH=/workspace/src:${PYTHONPATH}

WORKDIR /workspace
COPY src ./src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
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

RUN python3 --version \
    && python3 -m pip --version

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.1+cu118 \
    torchaudio==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip install --no-cache-dir \
    plotly \
    pyyaml \
    lietorch==0.6.2

WORKDIR /workspace/src

RUN git clone https://gitlab.inria.fr/bkerbl/simple-knn.git pose_gaussian/submodules/simple-knn \
    && git clone https://github.com/g-truc/glm.git pose_gaussian/submodules/self-xray-gaussian-rasterization-voxelization/third_party/glm

RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    scikit-learn==1.3.2

RUN python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

RUN python3 -m pip install --no-cache-dir \
    pose_gaussian/submodules/simple-knn \
    --no-build-isolation

RUN python3 -m pip install --no-cache-dir \
    pose_gaussian/submodules/self-xray-gaussian-rasterization-voxelization \
    --no-build-isolation

WORKDIR /workspace

RUN wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip \
    && unzip v2.3.zip \
    && python3 -m pip install --no-cache-dir TIGRE-2.3/Python --no-build-isolation \
    && rm v2.3.zip

WORKDIR /workspace/src

CMD ["/bin/bash"]