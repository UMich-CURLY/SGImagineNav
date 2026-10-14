# Local installation

These steps install the same versions as the Docker image. You'll need Linux on x86-64, an NVIDIA GPU (Ampere or newer), a driver compatible with CUDA 12.8 and Conda. Run the commands from the SGImagineNav repository root with the `sg_vln` environment active.

## Build tools and CUDA

On Ubuntu, install the build tools and graphics libraries:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential git curl ffmpeg patch unzip ninja-build \
  libegl1-mesa-dev libgl1-mesa-dev libglib2.0-0 libglm-dev libjpeg-dev \
  libx11-dev libxcb1 libxext6 libxrender1
```

Install [CUDA 12.8 from NVIDIA's Conda channel](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html#conda-installation), then install the Python dependencies. Use the system compiler so the native builds can find Ubuntu's graphics libraries:

```bash
conda install -c nvidia/label/cuda-12.8.0 cuda-toolkit=12.8.0 -y
export CUDA_HOME="$CONDA_PREFIX"
export CC="$(command -p -v gcc)" CXX="$(command -p -v g++)"
export TORCH_CUDA_ARCH_LIST="8.0+PTX;12.0"
export MAX_JOBS=8
pip install -r requirements.txt
mkdir -p third_party
```

## Habitat and PyTorch3D

Build Habitat-Sim with headless rendering, CUDA and Bullet, and install Habitat-Lab:

```bash
git clone --recursive --branch v0.3.3 \
  https://github.com/facebookresearch/habitat-sim.git third_party/habitat-sim
(cd third_party/habitat-sim && \
  python setup.py build_ext --parallel 8 install --headless --with-cuda --bullet)

git clone --branch v0.3.3 \
  https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
pip install --no-deps -e third_party/habitat-lab/habitat-lab

FORCE_CUDA=1 pip install --no-build-isolation \
  git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9
```

## GroundingDINO and SAM

We use GroundingDINO for object detection and SAM for segmentation. Install the pinned source with the included PyTorch compatibility patch:

```bash
export GSA_PATH="$PWD/third_party/Grounded-Segment-Anything"
git init "$GSA_PATH"
git -C "$GSA_PATH" remote add origin \
  https://github.com/IDEA-Research/Grounded-Segment-Anything.git
git -C "$GSA_PATH" fetch --depth 1 origin \
  126abe633ffe333e16e4a0a4e946bc1003caf757
git -C "$GSA_PATH" checkout FETCH_HEAD
patch -d "$GSA_PATH" -p1 < docker/groundingdino-torch27.patch

pip install --no-deps "$GSA_PATH/segment_anything"
pip install --no-deps --no-build-isolation \
  "$GSA_PATH/GroundingDINO"
```

Download the two checkpoints:

```bash
curl -fL https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -o "$GSA_PATH/sam_vit_h_4b8939.pth"
curl -fL https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -o "$GSA_PATH/groundingdino_swint_ogc.pth"
```

## Ollama

The evaluator starts its own Ollama workers. Install the binary in the checkout:

```bash
mkdir -p third_party/ollama
curl -fL https://github.com/ollama/ollama/releases/download/v0.6.8/ollama-linux-amd64.tgz \
  | tar -xz -C third_party/ollama
export PATH="$PWD/third_party/ollama/bin:$PATH"
```

## Check the installation

```bash
python -m pip check
python scripts/check_installation.py
```

You're ready to [prepare HM3D and run evaluation](../README.md#prepare-the-dataset). When opening a new terminal, use the local setup commands in the README to activate the environment and set its paths. `YOLO_CONFIG_DIR` keeps Ultralytics settings in this checkout.
