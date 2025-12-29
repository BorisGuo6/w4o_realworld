# Pipeline

1. Input (image, depth)    # camera save to path
2. Segmentation (image, depth, text) → Output (mask)   # GroundedSAM save to path
3. Graspnet_v2 (image, depth,  mask) → Output (grasp_pose)   # In docker, save to path
4. World_model (image, depth) → Output (goal_image)
5. Semantic_matching (image, depth, goal_image) → Output (transformation)
6. Calculate (grasp_pose, transformation) → Output (goal_pose)
7. Motion_planning (initial_pose, grasp_pose) → Output (action_sequence)  # cuRobo
8. Motion_planning (grasp_pose, goal_pose) → Output (action_sequence)  # cuRobo



## 0. Calibration
See `~/w4o/rwVR/rel/calib/calibration.md`

In `~/w4o`:
```bash
conda activate camcalib
...
```

## 1. Input (image, depth)
```bash
conda activate w4o2
python ~/w4o/rwVR/rel/cameras/orbbec.py
```
 - pdb mode: 'c': capture current frame
 - save [image, depth, pointcloud] to $pwd/data/raw_data/
 - By default, subfolder is named as the current time

## 2.  Segmentation (image, depth, text) → Output (mask)
Change `base_name` inside `grounded_sam`. 
(By default, `get_newest()` provides the newest capture)
If object changes, modify the prompt object name.
```bash
conda activate w4o2
python ~/w4o/World4Omni_rw/grounded_sam.py
```

## 3. Grasping (image, depth,  mask) → Output (grasp_pose)
Change the file name and hyperparameters in `sample_based_grasp.py `. 
(By default, `get_newest()` provides the newest capture)
```bash
conda activate w4o2
python World4Omni_rw/sample_based_grasp.py 
python World4Omni_rw/exec_grasp.py
```

## 4. World_model (image, text) → Output (goal_image)
```bash
export PYTHONNOUSERSITE=1
export GEMINI_API_KEY=AIzaSyCqX_zpTdFzrNt2IqejA1FWwa6_jxX11do

conda activate world
python World_model/test/test_world_model.py
```

## 4.5 mask goal
```bash
conda activate w4o2
python ~/w4o/World4Omni_rw/grounded_sam.py  goal
```


## 5. Semantic_matching (image, image2, depth1, mask1, mask2) → Output (transformation)
Download https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth, put it in feature_based_icp/

Solve the issue: RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR
in `feature_based_icp/third_party/ODISE/odise/modeling/meta_arch/ldm.py #L436`, set `self.freeu=False`.
See https://github.com/Junyi42/GeoAware-SC/issues/1.

run:
```bash
bash feature_based_icp/transformation.sh 0 [data_dir] real
```

## 6. Calculate (grasp_pose, transformation) → Output (goal_pose)
```bash
conda activate w4o2
python World4Omni_rw/exec_goal.py
```

## 7. Motion_planning (initial_pose, grasp_pose, goal_pose) → Output (action_sequence)
```bash
conda activate w4o2
python World4Omni_rw/execute_xxx.py
```

(below is not used)
```bash
conda activate curobo
python xarmMotionPlanner-main/xarmMontionPlanner.py 
```


## 8. record video:
Start another terminal. run:
```bash
conda activate camcalib
python ~/w4o/rwVR/rel/cameras/orbbec_record.py
```