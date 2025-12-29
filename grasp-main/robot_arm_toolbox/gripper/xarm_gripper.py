import numpy as np
import time
import sys
from xarm.wrapper import XArmAPI 

class XarmGripper(object):
    def __init__(self, arm):
        self.arm = arm
        code = arm.set_gripper_mode(0)
 
        print('set gripper mode: location mode, code={}'.format(code))

        code = arm.set_gripper_enable(True)
        print('set gripper enable, code={}'.format(code))

        code = arm.set_gripper_speed(5000)
        print('set gripper speed, code={}'.format(code))

        code = arm.set_gripper_position(850, wait=True)
        print('[wait]set gripper pos, code={}'.format(code))


    def gripper_action(self, width=850, sleep_time=0.2, wait=True):
        self.arm.set_gripper_position(width, wait=wait)
        time.sleep(sleep_time)

    def get_gripper_position(self):
        code, position = self.arm.get_gripper_position()
        return position

    def close_gripper(self, sleep_time=0.2, wait=True):
        self.arm.set_gripper_position(0, wait=wait)
        time.sleep(sleep_time)