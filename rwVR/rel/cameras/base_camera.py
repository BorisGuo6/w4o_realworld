import cv2
from pathlib import Path
from loguru import logger as lgr
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from abc import abstractmethod


class Camera:
    def __init__(self, serial_number=None):
        self.serial_number = serial_number
        self.pipeline = None     
    
    @abstractmethod
    def set_intrinsics(self, fx, fy, ppx, ppy):
        pass
    
    @abstractmethod
    def getCurrentData(self, pointcloud=True):
        pass
    
    def stop(self):
        self.pipeline.stop()
        


if __name__ == "__main__":

    get_realsense_serial_numbers()
    
