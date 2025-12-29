Execute in sequence
000_extrinsic_init.py   # get a initial pose of camera
001_capture_data_manual.py  # use manual mode of the robot arm (no gripper), and move it randomly.
002_sam_robot_arms.py   # manually sam the robot arm
003_camera_pose_opt.py  # automatically optimize the camera pose
(optional) 004_drag_camera_pose.py  # manually drag to align the camera

