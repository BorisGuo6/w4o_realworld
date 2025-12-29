
### capture the rgb image and depth image and PC from orbbec camera,
## Note: press C to capture, the captured imgs will be saved in the current dir

```python
python orbbec.py # /home/chn-4o/gpt-4o/rwVR/rel/cameras/
```
### if you want to record a video, you can change the code in the `orbbec.py` to uncomment line 446 - 448.


### Run rw exp
```python
python graspnet_rw.py # /home/chn-4o/gpt-4o/World4Omni/
```

# Usage
Stage 1: GraspNet
You need to change the prompt for the SAM input (line 219-238)
Stage 2: transformation
You need to load your task transformation by modify the file_path (line 247-249)

If you want to record a video from the camera while the arm is moving, you need to saperately run `graspnet_rw.py` and `python graspnet_rw_move_arm_only.py`

You can first run `graspnet_rw.py` to compute `ee_pose_in`, `ee_pose_out`, `t_goal`, `euler_goal`, and copy them to `python graspnet_rw_move_arm_only.py` and run it.

