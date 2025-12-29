# Feature-based ICP

## Environment Setup

To install the required dependencies, use the following commands:

```bash
conda create -n geo-aware python=3.9
conda activate geo-aware
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev
git clone git@github.com:ztr583/feature_based_icp.git
cd feature_based_icp
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/usr/local/cuda' | paste -sd:)
pip install -e .
```

PS: There are some common issues when installing Mask2Former. You might find [this issue](https://github.com/Junyi42/sd-dino/issues/11) helpful if you encounter any problems.

(Optional) You may want to install [xformers](https://github.com/facebookresearch/xformers) for efficient transformer implementation (which can significantly reduce the VRAM consumpution):

```
pip install xformers==0.0.16
```

(Optional) You may also want to install [SAM](https://github.com/facebookresearch/segment-anything) to extract the instance masks for adaptive pose alignment technique:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Get Started
<!-- Put the data in pcd_data directory:
```
--<your/img/path>--1.png
                 |-2.png
                 |-...
                 |-1_mask.png
                 |-2_mask.png
                 |-...
``` -->
```
--<your/img/path>--rgb_init.png
                 |-rgb_goal.png
                 |-...
                 |-mask_init.png
                 |-mask_goal.png
                 |-...
                 |-depth_init.npy
``` 

<!-- ### Shell-based All-in-one Running
```bash
sh transformation.sh <GPU_index> <your/image/path> <your/save/path>
``` -->
### Shell-based All-in-one Running
```bash
sh transformation.sh <GPU_index> <your/image/path> <sim/or/real>
```
All results will be save to image path.
For real world, `GPU_index=0`.

### Obtaining Metric Depth
#### Using DepthAnything (Preferred)
Run
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2/metric_depth
conda create -n depthanything python=3.10
conda activate depthanything
pip install -r requirements.txt
```
Put the `depth_anything_v2_metric_hypersim_vitl.pth` under `feature_based_icp/checkpoints/`

In feature_based_icp/, Run
```bash
python generate_metric_depth.py --img-path <your/image/path>
```

#### Using FoundationStero
Put the left and right images in pcd_data directory:
```bash
--<your/img/path>--1.png
                 |-2_left.png
                 |-2_right.png
                 |-...
                 |-1_mask.png
                 |-2_mask.png
                 |-...
                 |-depth_*.npy ## real depth
```

Run
```bash
git clone https://github.com/iKrishneel/foundation_stereo.git
cd FoundationStereo-AGX
conda create -n foundation_stereo python=3.9
pip install -r requirements.txt
conda activate foundation_stereo
```
Following the instruction in FoundationStereo, put the pretrained weight in the directory. Put the generate_disp.sh file in FoundationStereo-AGX directory, and change the corresponding directory in the bash file.

Run
```bash
sh generate_disp.sh
```

### Extrating/Visualization the post-processed features

Run
```bash
conda activate geo-aware
python matching_generation.py
python ft_based_umeyama.py
```

The final transformation will be stored in the \tt{transformation.pkl}, including the rotation, transition and scale between 2 predicted depth.

Run
```bash
python depth_gen.py
```
The visualization of pointwise matching will be displayed.