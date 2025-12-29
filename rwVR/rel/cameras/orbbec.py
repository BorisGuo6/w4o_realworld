# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import time
import datetime
import os
import imageio
from typing import Union, Any, Optional

import cv2
import numpy as np
from queue import Queue
import sys
sys.path.insert(0, "/home/chn-4o/gpt-4o/pyorbbecsdk")
from pyorbbecsdk import Config, Context, OBLogLevel
from pyorbbecsdk import OBSensorType
from pyorbbecsdk import Pipeline
from pyorbbecsdk import PointCloudFilter, AlignFilter
from pyorbbecsdk import OBFormat
from pyorbbecsdk import OBCalibrationParam
from pyorbbecsdk import OBStreamType

from pyorbbecsdk import FormatConvertFilter, VideoFrame
from pyorbbecsdk import OBFormat, OBConvertFormat

from pyorbbecsdk import *

# from .base_camera import Camera
from rel.cameras.base_camera import Camera
import open3d as o3d


ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 1800  # 10000mm

last_depth_frame = None



def crop_by_radius(points: np.ndarray,
                   center: np.ndarray = np.zeros(3),
                   max_dist: float = 1.0) -> np.ndarray:
    """
    Args:
      points   : (N,3) 点云坐标
      center   : (3,) 参考点，默认原点
      max_dist : 最大保留半径（m）
    Returns:
      cropped  : (M,3) 距离 center 小于 max_dist 的点
    """
    # 计算每个点到 center 的欧氏距离
    dists = np.linalg.norm(points - center[None, :], axis=1)
    # 根据距离 threshold 筛选
    mask = dists < max_dist
    return points[mask]


def on_new_frame_callback(frames):
    global last_depth_frame
    depth_frame = frames.get_depth_frame()
    last_depth_frame = depth_frame
        
        
def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image




class Orbbec(Camera):
    def __init__(self, serial_number=None, use_depth=True, use_color=True):
        super().__init__(serial_number)

        context = Context()
        self.pipeline = Pipeline()
        self.config = Config()
        Context().set_logger_level(OBLogLevel.NONE)
        
        # "You can only use depth or color, not both"
        self.use_depth = use_depth
        self.use_color = use_color
        # assert (self.use_depth and not self.use_color) or (self.use_color and not self.use_depth)
        
        if self.use_depth:
            try:
                profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                assert profile_list is not None
                depth_profile = profile_list.get_default_video_stream_profile()
                assert depth_profile is not None
                print("depth profile: ", depth_profile)
                self.config.enable_stream(depth_profile)
            except Exception as e:
                print(e)
                return
            
        if self.use_color:
            try:
                profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                assert profile_list is not None
                color_profile = profile_list.get_default_video_stream_profile()
                assert color_profile is not None
                print("color profile: ", color_profile)
                self.config.enable_stream(color_profile)
            except Exception as e:
                print(e)
        
        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)
        # self.pipeline.start(self.config, lambda frame_set: on_new_frame_callback(frame_set))
    
        camera_param = self.pipeline.get_camera_param()
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.point_cloud_filter = PointCloudFilter()
        self.point_cloud_filter.set_camera_param(camera_param)
        
        camera_pram = self.pipeline.get_camera_param()
        # if self.use_color:
        self.w = camera_pram.rgb_intrinsic.width
        self.h = camera_pram.rgb_intrinsic.height
        self.K = np.array(
            [
                [camera_pram.rgb_intrinsic.fx, 0, camera_pram.rgb_intrinsic.cx],
                [0, camera_pram.rgb_intrinsic.fy, camera_pram.rgb_intrinsic.cy],
                [0, 0, 1],
            ]
        )

        self.fov_x = 2 * np.arctan(
            self.w / (2 * camera_pram.rgb_intrinsic.fx)
        )  # Horizontal FOV in radians
        self.fov_y = 2 * np.arctan(
            self.h / (2 * camera_pram.rgb_intrinsic.fy)
        )  # Vertical FOV in radians
        self.aspect_ratio = self.w / self.h
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.w,
            self.h,
            camera_pram.rgb_intrinsic.fx,
            camera_pram.rgb_intrinsic.fy,
            camera_pram.rgb_intrinsic.cx,
            camera_pram.rgb_intrinsic.cy,
        )
        
    def set_intrinsics(self, fx, fy, ppx, ppy):
        # TODO: set intrinsics in orbbec camera
        self.fx = fx
        self.fy = fy
        
        self.ppx = ppx
        self.ppy = ppy
        self.K = np.array(
            [
                [fx, 0, ppx],
                [0, fy, ppy],
                [0, 0, 1],
            ]
        )
        self.fov_x = 2 * np.arctan(self.w / (2 * self.fx))
        self.fov_y = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect_ratio = self.w / self.h
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.w,
            self.h,
            fx,
            fy,
            ppx,
            ppy,
        )
        
        
    def getCurrentData(self, pointcloud=True):
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            # frames = None 
            
            self.rgb_color_image = None
            self.depth_data = None
            if self.use_color and self.use_depth:
                try:
                    self.color_frame = frames.get_color_frame()
                    self.depth_frame = frames.get_depth_frame()

                    if not self.color_frame or not self.depth_frame:
                        continue
                    
                    # # shape
                    # print(f"color_frame_shape: {self.color_frame.get_width()}, {self.color_frame.get_height()}")
                    # print(f"depth_frame_shape: {self.depth_frame.get_width()}, {self.depth_frame.get_height()}")
                    
                    frames = self.align_filter.process(frames)
                    if not frames:
                        continue
                    frames  = frames.as_frame_set()
                    self.color_frame = frames.get_color_frame()
                    self.depth_frame = frames.get_depth_frame()

                    # # shape
                    # print(f"color_frame_shape: {self.color_frame.get_width()}, {self.color_frame.get_height()}")
                    # print(f"depth_frame_shape: {self.depth_frame.get_width()}, {self.depth_frame.get_height()}")

                    if not self.color_frame or not self.depth_frame:
                        continue

                    bgr_color_image = frame_to_bgr_image(self.color_frame)
                    if bgr_color_image is None:
                        print("Failed to convert frame to image")
                        continue

                    self.rgb_color_image = cv2.cvtColor(bgr_color_image, cv2.COLOR_BGR2RGB)
                    try:
                        depth_data = np.frombuffer(self.depth_frame.get_data(), dtype=np.uint16).reshape(
                            (self.depth_frame.get_height(), self.depth_frame.get_width()))
                    except ValueError:
                        print("Failed to reshape depth data")
                        continue
                    depth_data = depth_data.astype(np.float32) * self.depth_frame.get_depth_scale()
                    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)

                    self.depth_data = depth_data / 1000 # mm to meters

                except KeyboardInterrupt:
                    break
            elif self.use_depth and not self.use_color:
                # if last_depth_frame is not None:
                #     self.depth_frame = last_depth_frame
                # else: 
                #     continue
                self.depth_frame = frames.get_depth_frame()
                print(f"depth_frame={self.depth_frame.get_timestamp()}")
                
                if self.depth_frame is None:
                    continue
                width = self.depth_frame.get_width()
                height = self.depth_frame.get_height()
                scale = self.depth_frame.get_depth_scale()

                depth_data = np.frombuffer(self.depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))

                depth_data = depth_data.astype(np.float32) * scale
                depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
                self.depth_data = depth_data / 1000 # mm to meters
            elif self.use_color and not self.use_depth:
                self.color_frame = frames.get_color_frame()
                if self.color_frame is None:
                    continue
                # covert to RGB format
                bgr_color_image = frame_to_bgr_image(self.color_frame)
                if bgr_color_image is None:
                    print("failed to convert frame to image")
                    continue
                self.rgb_color_image = cv2.cvtColor(bgr_color_image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("You must use depth or color")
            
            rtr_dict = {}
            rtr_dict["rgb"] = self.rgb_color_image
            rtr_dict["depth"] = self.depth_data
            rtr_dict["colored_depth"] = None
            rtr_dict["pointcloud_o3d"] = None
            rtr_dict["pointcloud_np"] = None
            rtr_dict["points_colors"] = None
            
            if pointcloud and self.use_depth and self.use_color:
                frame = self.align_filter.process(frames)
                scale = self.depth_frame.get_depth_scale()
                self.point_cloud_filter.set_position_data_scaled(scale)

                self.point_cloud_filter.set_create_point_format(
                    OBFormat.RGB_POINT if self.color_frame is not None else OBFormat.POINT)
                point_cloud_frame = self.point_cloud_filter.process(frame)
                self.points = self.point_cloud_filter.calculate(point_cloud_frame)
                self.points_np = np.array(self.points[:, :3]) / 1000.0
                self.points_colors = np.array(self.points[:, 3:6], dtype=np.uint8)
                
                rtr_dict["points_colors"] = self.points_colors
                rtr_dict["pointcloud_np"] = self.points_np
                
            if pointcloud and self.use_depth and not self.use_color:
                scale = self.depth_frame.get_depth_scale()
                self.point_cloud_filter.set_position_data_scaled(scale)
                self.point_cloud_filter.set_create_point_format(OBFormat.POINT)
                point_cloud_frame = self.point_cloud_filter.process(self.depth_frame)
                self.points = self.point_cloud_filter.calculate(point_cloud_frame)
                self.points_np = np.array(self.points[:, :3]) / 1000.0
                
                # rtr_dict["pointcloud_o3d"] = self.points / 1000 # pointcloud_o3d has not been scaled.
                rtr_dict["pointcloud_np"] = self.points_np
            return rtr_dict
            
            # frames = self.pipeline.wait_for_frames(100)
            # if frames is None:
            #     print("No frames received.")
            #     continue
            
            # self.depth_frame = frames.get_depth_frame()
            # self.color_frame = frames.get_color_frame()
            
            # if self.depth_frame is None or self.color_frame is None:
            #     print("No depth_frame or color_frame received.")
            #     continue
            
            # rtr_dict = {}
            # rtr_dict["rgb"] = self.color_frame
            # rtr_dict["depth"] = self.depth_frame
            # rtr_dict["colored_depth"] = None
            # rtr_dict["pointcloud_o3d"] = None
            # rtr_dict["pointcloud_np"] = None
            
            # if pointcloud:
            #     scale = self.depth_frame.get_depth_scale()
            #     self.point_cloud_filter.set_position_data_scaled(scale)
            #     self.point_cloud_filter.set_create_point_format(OBFormat.POINT)
            #     point_cloud_frame = self.point_cloud_filter.process(self.depth_frame)
            #     self.points = self.point_cloud_filter.calculate(point_cloud_frame)
            #     self.points_np = np.array(self.points) / 1000.0
                
            #     rtr_dict["pointcloud_o3d"] = self.points / 1000 # pointcloud_o3d has not been scaled.
            #     rtr_dict["pointcloud_np"] = self.points_np
            # return rtr_dict


# def record_rgb_video(camera: Orbbec, fps=30, output_dir="rgb_video"):
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Get current timestamp
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Initialize video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(f"{output_dir}/rgb_{timestamp}.mp4", fourcc, fps, (1920, 1080))
    
#     print(f"recording rgb video..., please press Ctrl+C to stop")
#     while True:
#         try:
#             rgb = camera.getCurrentData(pointcloud=False)['rgb']
#             if rgb is not None:
#                 print(f"rgb={rgb.shape}")
#                 video_writer.write(rgb)
#         except KeyboardInterrupt:
#             break
    
#     print(f"stopping rgb video recording...")
#     video_writer.release()

def record_video_imageio(camera: Orbbec, fps=30, output_dir="rgb_video"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/rgb_{timestamp}.mp4"

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

    print(f"To start recording, please press Enter")
    input()
    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            rgb = camera.getCurrentData(pointcloud=False)['rgb']
            if rgb is not None:
                # Convert to uint8 if needed
                if rgb.dtype != np.uint8:
                    print(f"rgb.dtype={rgb.dtype}")
                    rgb = (rgb * 255).astype(np.uint8)
                writer.append_data(rgb)
            time.sleep(1 / fps)
    except KeyboardInterrupt:
        print("Stopped recording.")
    finally:
        writer.close()



if __name__ == "__main__":
    ob = Orbbec(use_depth=False, use_color=True)
    record_video_imageio(camera=ob, fps=30, output_dir="rgb_video")
    exit()
    import viser
    server = viser.ViserServer()
    
    ob = Orbbec(use_depth=True, use_color=True)
    time.sleep(3)
    
    while True:
        try:
            breakpoint()
            tick = time.time()
            rtr_dict = ob.getCurrentData(pointcloud=True)
            # points = ob.getCurrentData(pointcloud=True)
            # if points is None:
            #     continue
            
            # # print("pc_points", points.shape)
            # server.scene.add_point_cloud(
            #     "pc",
            #     points,
            #     colors=(255, 0, 0),
            #     point_size=0.0005,
            # )
            # tok = time.time() - tick
            # print(f"fps: {1 / tok:.2f}")
            if rtr_dict is None:
                print(f"1")
                continue
            pc_points = rtr_dict["pointcloud_np"]
            # pc_colors = np.array(np.array([1, 0, 0]) * 255, np.uint8)
            pc_colors = rtr_dict["points_colors"]

            rgb = rtr_dict['rgb'] # (1080, 1920, 3)
            depth = rtr_dict['depth'] # (1080, 1920)
            print(f"rgb={rgb.shape}, depth={depth.shape}")

            # crop to (1080, 1440)
            if rgb is not None:
                h, w = rgb.shape[:2]
                start_x = 0  # Center the crop horizontally
                rgb = rgb[:, start_x:start_x + 1440]  # Crop to 1440 width
                print(f"Cropped rgb shape: {rgb.shape}")
            
            # crop to (1080, 1440)
            if depth is not None:
                h, w = depth.shape[:2]
                start_x = 0  # Use same cropping as RGB
                depth = depth[:, start_x:start_x + 1440]  # Crop to 1440 width
                print(f"Cropped depth shape: {depth.shape}")

            # save img
            import cv2
            
            # Create timestamp for filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save RGB image
            if rgb is not None:
                rgb_filename = f"rgb_{timestamp}.png"
                cv2.imwrite(rgb_filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                print(f"Saved RGB image to {rgb_filename}")

            # save depth as .npy
            if depth is not None:
                depth_filename = f"depth_{timestamp}.npy"
                np.save(depth_filename, depth)
                print(f"Saved depth data to {depth_filename}")
            
            # Save point cloud data
            if pc_points is not None and pc_colors is not None:
                pc_filename = f"pointcloud_{timestamp}"
                np.save(f"{pc_filename}_points.npy", pc_points)
                np.save(f"{pc_filename}_colors.npy", pc_colors)
                print(f"Saved point cloud data to {pc_filename}_points.npy and {pc_filename}_colors.npy")

            # crop pc_points by distance
            croped_pc = crop_by_radius(points = pc_points, center=np.zeros(3), max_dist=1.7)

            # downsample pc
            # Desired number of points
            # M = 1000000
            M = pc_points.shape[0]
            N = pc_points.shape[0]
            indices = np.random.choice(N, size=min(M, N), replace=False)
            down_pc_points = pc_points[indices, :]
            down_pc_colors = pc_colors[indices, :]

            server.scene.add_point_cloud(
                "pc",
                down_pc_points,
                down_pc_colors, 
                point_size=0.00001,
                point_shape="square"
            )
            
            tok = time.time() - tick
            # print(f"fps: {1 / tok:.2f}")

            # save pc

            


            
        except KeyboardInterrupt:
            break
        
    ob.stop()
