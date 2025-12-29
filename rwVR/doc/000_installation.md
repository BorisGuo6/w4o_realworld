

# Installation 

Install the 

```python 
pip install -e . 
```

# Realsense Requirements 

```python 
pip install loguru numpy open3d pyrealsense2 scipy opencv-python xarm-python-sdk
```

# Orbbec Requirements
Refer to the [official website](https://github.com/orbbec/pyorbbecsdk) for installation.

# Extrinsics Calibration Requirements 
```python 
# install torch 
pip install torch pytorch-kinematics trimesh
```

## Install nvdiffrast
```python 
pip install ninja 
pip install git+https://github.com/NVlabs/nvdiffrast
```

## SAM Requirements
```python
pip install git+https://github.com/facebookresearch/segment-anything.git
```
First download a model checkpoint.

## FoundationPose

Tested on python3.9+pytorch200+cuda11.8

```
git clone https://github.com/NVlabs/FoundationPose.git


conda install conda-forge::eigen=3.4.0

# NOTE: change the /eigen/path/under/conda here to something like /home/xzx/anaconda3/envs/dro/include/eigen3
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# NOTE: There's a bug in the original repo. You actually need to add something like /home/xzx/anaconda3/envs/dro/include/eigen3 to `include_dirs` in FoundationPose/bundlesdf/mycuda/setup.py
# If the following lines have errors: you may need:
# sudo apt update
# sudo apt install libboost-system-dev libboost-program-options-dev

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

# You should see: Successfully installed common
# Then download the weights and demo data as instructed in the repo's readme. And `python run_demo.py`, you should see a video demo. 
# Add __init__.py to FoundationPose
```
