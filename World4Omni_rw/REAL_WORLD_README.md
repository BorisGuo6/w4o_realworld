# Low-level Policy for Real World Experiments

## Environment Setup
Our environment follows https://github.com/graspnet/graspnet-baseline.

Get the code.
```bash
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```
We also require transformers for Grounding SAM.
```bash
pip install --upgrade -q git+https://github.com/huggingface/transformers
```

## Usage
Download the pretrained weights can be downloaded from:
- `checkpoint-rs.tar`
[[Google Drive](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing)]
[[Baidu Pan](https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A)]

For usage, you can load the model with the following code:
```python
from model_real_world import GroundedGraspNet

graspnet = GroundedGraspNet(ckpt_path='PATH_TO_CHECKPOINT')
prompt = ['the red cup.']
action = graspnet.step(obs, prompt, cam_int, cam2world)
```
For real world deployment, you need to modify the `process_input` part and implement the grasp selection logics.

The output of the model is a list of grasp candidates, each with a score. The higher the score, the better the grasp. You can select the top-k candidates.

The grasp's pose is in the camera coordinate system. You need to transform it to the robot's coordinate system.
