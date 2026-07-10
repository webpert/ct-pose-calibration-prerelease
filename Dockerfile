FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

WORKDIR /workspace
COPY src ./src

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip build-essential cmake ninja-build \
    libgl1-mesa-glx libglib2.0-0 libglew-dev libassimp-dev \
    libboost-all-dev libgtk-3-dev libopencv-dev \
    python3 python3-pip python-is-python3 tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# CT POSE Gaussian
WORKDIR /workspace/src

RUN git clone https://gitlab.inria.fr/bkerbl/simple-knn.git pose_gaussian/submodules/simple-knn
RUN git clone https://github.com/g-truc/glm.git pose_gaussian/submodules/self-xray-gaussian-rasterization-voxelization/third_party/glm

RUN pip install -r requirements.txt

RUN python -c "import torch; print(torch.__version__, torch.version.cuda)"

RUN python -m pip install \
    pose_gaussian/submodules/simple-knn \
    --no-build-isolation

RUN python -m pip install \
    pose_gaussian/submodules/self-xray-gaussian-rasterization-voxelization \
    --no-build-isolation

# Lie torch
WORKDIR /workspace/src
RUN git clone https://github.com/princeton-vl/lietorch.git
WORKDIR /workspace/src/lietorch
RUN python -m pip install -U numpy ninja setuptools wheel && \
    git submodule sync --recursive && \
    git submodule update --init --recursive && \
    rm -rf build dist *.egg-info && \
    find . -name "*.so" -delete && \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" MAX_JOBS=4 python -m pip install -e . --no-build-isolation

# TIGRE
WORKDIR /workspace

RUN wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip && \
    unzip v2.3.zip && \
    pip install TIGRE-2.3/Python --no-build-isolation && \
    rm v2.3.zip

WORKDIR /workspace/src

CMD ["/bin/bash"]
