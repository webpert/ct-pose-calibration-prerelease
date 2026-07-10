FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
ENV PYTHONPATH=/workspace/src:${PYTHONPATH}

WORKDIR /workspace
COPY src ./src

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip build-essential cmake ninja-build \
    libgl1-mesa-glx libglib2.0-0 libglew-dev libassimp-dev \
    libboost-all-dev libgtk-3-dev libopencv-dev \
    libxrender1 libxext6 libsm6 libgomp1 \
    python3 python3-pip python3-venv python3-dev python-is-python3 tzdata ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN which python3 && python3 --version && python3 -m pip --version

RUN ln -fs /usr/bin/python3 /usr/bin/python \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

RUN python3 -m pip install --no-cache-dir torch==2.0.2 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip install --no-cache-dir plotly pyyaml lietorch==0.6.2

# Pose Gaussian
WORKDIR /workspace/src

RUN git clone https://gitlab.inria.fr/bkerbl/simple-knn.git pose_gaussian/submodules/simple-knn
RUN git clone https://github.com/g-truc/glm.git pose_gaussian/submodules/self-xray-gaussian-rasterization-voxelization/third_party/glm

RUN python3 -m pip install --no-cache-dir -r requirements.txt scikit-learn==1.3.2

RUN python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

RUN python3 -m pip install \
    pose_gaussian/submodules/simple-knn \
    --no-build-isolation

RUN python3 -m pip install \
    pose_gaussian/submodules/self-xray-gaussian-rasterization-voxelization \
    --no-build-isolation

# TIGRE
WORKDIR /workspace

RUN wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip && \
    unzip v2.3.zip && \
    python3 -m pip install --no-cache-dir TIGRE-2.3/Python --no-build-isolation && \
    rm v2.3.zip

WORKDIR /workspace/src

CMD ["/bin/bash"]