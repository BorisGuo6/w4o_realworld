# from realman_toolbox.realman import RealManRobot
# from realman_toolbox.realman_api import RobotStatus
# arm = RealManRobot("RM75", '192.168.1.18')
# RealtimePush_Callback = arm.robot.Realtime_Arm_Joint_State(RobotStatus)
# print(RealtimePush_Callback.joint_status)
# print(target_joint.temperature, target_joint.voltage, target_joint.current, target_joint.en_state, target_joint.err_flag, target_joint.sys_err)
# # target_joint[0] = target_joint[0] + 100
# # arm.robot.Movej_Cmd(target_joint, v = 1, r = 0)

from Robotic_Arm.rm_robot_interface import *

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
print(handle.id)
i = 0
# print(arm.rm_get_current_arm_state())
while i < 50:
    arm.rm_movej([10.25600001215934753, 31.784000396728516, -0.3799999952316284, 48.715999603271484, 10.10400000214576721, 119.58399963378906, -10.061000000685453415], 2, 0, 0, 1)
    print(arm.rm_get_current_arm_state())
    # arm.rm_set_gripper_position(2, block=True, timeout=2)
    arm.rm_movej([0.25600001215934753, 11.784000396728516, -50.3799999952316284, 58.715999603271484, 0.10400000214576721, 109.58399963378906, -0.061000000685453415], 2, 0, 0, 1)
    print(arm.rm_get_current_arm_state())
    # arm.rm_set_gripper_position(900, block=True, timeout=2)
    print('\n', i)
    i = i + 1
arm.rm_delete_robot_arm()

# print(arm.rm_get_current_arm_state())
# arm.rm_movel([0.175669, -0.162478, 0.415667, -2.984, 0.07, 2.29], 20, 0, 0, 1)
# print(arm.rm_get_current_arm_state())
# arm.rm_set_gripper_position(2, block=True, timeout=2)
# arm.rm_movel([0.27779, 0.073681, 0.343404, 2.903, -0.306, -2.698], 20, 0, 0, 1)
# print(arm.rm_get_current_arm_state())
# arm.rm_set_gripper_position(900, block=True, timeout=2)