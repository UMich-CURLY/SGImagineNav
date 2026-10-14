FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GSA_PATH=/dependencies/Grounded-Segment-Anything \
    TORCH_CUDA_ARCH_LIST="8.0+PTX;12.0" \
    MAX_JOBS=8 \
    FORCE_CUDA=1 \
    AM_I_DOCKER=True \
    BUILD_WITH_CUDA=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential bzip2 ca-certificates cmake curl ffmpeg git libegl1-mesa-dev \
    libgl1-mesa-dev libglib2.0-0 libglm-dev libjpeg-dev libx11-dev libxcb1 \
    libxext6 libxrender1 ninja-build patch unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/2.0.5 \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba create -y -p /opt/sg-vln -c conda-forge python=3.10.17 pip=25.0.1 \
    && micromamba clean -ay
ENV PATH=/opt/sg-vln/bin:${PATH}

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /dependencies
RUN git clone --recursive --branch v0.3.3 https://github.com/facebookresearch/habitat-sim.git \
    && cd habitat-sim \
    && python setup.py build_ext --parallel 8 install --headless --with-cuda --bullet
RUN git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-lab.git \
    && pip install --no-deps -e habitat-lab/habitat-lab
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9


# GroundingDINO PR #415 updates the tensor API used by CUDA dispatch.
COPY docker/groundingdino-torch27.patch /tmp/groundingdino-torch27.patch
RUN git init Grounded-Segment-Anything \
    && cd Grounded-Segment-Anything \
    && git remote add origin https://github.com/IDEA-Research/Grounded-Segment-Anything.git \
    && git fetch --depth 1 origin 126abe633ffe333e16e4a0a4e946bc1003caf757 \
    && git checkout FETCH_HEAD \
    && patch -p1 < /tmp/groundingdino-torch27.patch \
    && pip install --no-cache-dir --no-deps ./segment_anything \
    && pip install --no-cache-dir --no-deps --no-build-isolation ./GroundingDINO \
    && curl -fL https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
       -o sam_vit_h_4b8939.pth \
    && curl -fL https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
       -o groundingdino_swint_ogc.pth
RUN curl -fL https://github.com/ollama/ollama/releases/download/v0.6.8/ollama-linux-amd64.tgz \
    | tar -xz -C /usr

COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /workspace/SG-VLN
CMD ["/bin/bash"]
