from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import sys
from openai import OpenAI
from vlm_plan.gptutils import *
import json
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles
from scipy.spatial import ConvexHull
import numpy as np
sys.path.append(os.getcwd())
from Env_Config.Robot.Franka import Franka
from Env_Config.Robot.Bimanual_Franka import Bimanual_Franka
from Env_Config.Gripper_Grasp.grasp_interface import grasp_checker, grasp_view

from Env_StandAlone.real.real_edge import Demo_Scene_Env
from Env_StandAlone.NP_manipulation import pixels_to_world
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Env_Config.Utils_Project.Transforms import Rotation
from VLM.target_indicator import pose_rotate_gen, pose_push_gen, choose_pose, pose_rotate_down_gen, pose_move_gen
from VLM.arkpoint import get_2d_point
from pdb import set_trace as bp
# from realworld.rw import get_image_depth
import time
# # 设置代理
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

api_key = "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"

client = OpenAI(api_key=api_key)


def split_actions(plan):
    plan = plan.strip().replace('```json', '').replace('```', '').strip()
    
    actions = json.loads(plan)

    subtasks = []

    for action in actions:
        subtasks.append(action)

    return subtasks



def plan(task = None, obs_image = None, prompt_plan = None, prompt_action = None):
    # plan = """
    #     [
    #         {
    #             "action": "push",
    #             "parameters": {
    #             "object": "keyboard",
    #             "place": "transparent target"
    #             }
    #         }
    #     ]
    # """

    plan = """
        [
            {
                "action": "push",
                "parameters": {
                "object": "box",
                "place": "table edge"
                }
            },
            {
                "action": "grasp",
                "parameters": {
                "object": "box"
                }
            },
            {
                "action": "move to",
                "parameters": {
                "place": "woolden stage"
                }
            },
            {
                "action": "release",
                "parameters": {}
            }
        ]
            """

    # plan = call_gpt_model(
    #     client=client,
    #     prompt1=prompt_plan,
    #     prompt2=prompt_action,
    #     task=task,
    #     images= [obs_image],
    # )

    print(plan)

    return plan



def get_env(task, obs_image, prompt1):
    
    env = call_gpt_model(
        client=client,
        prompt1=prompt1,
        task=task,
        images= [obs_image],
    )

    print(env)

    return env



def replan(env, subtask, following_subtasks, topdown_img, front_img, prompts, grasp_pose=False, ik=False):
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
    # set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False)
    for i in range(25):
        env.step()
    env.front_camera.get_rgb_graph(save_or_not=True, save_path="./front.png")
    env.top_camera.get_rgb_graph(save_or_not=True, save_path="./topdown.png")
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
    # set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=True)
    for i in range(25):
        env.step()
    get_err = False
    while not get_err :
        if not grasp_pose:
            prompt = prompts['check_nopose']
        elif not ik:
            prompt = prompts['check_ik']

        error_message = call_gpt_model(
            client=client,
            prompt1=prompt,
            task=subtask,
            images= [topdown_img, front_img],
        )
        print(error_message)
        error_message = error_message.strip().replace('```json', '').replace('```', '').strip()
        
        try:
            error = json.loads(error_message)
            error_type = error['error_type']
            get_err = True
        except json.JSONDecodeError as e:
            print(f"没有检查出error")
            continue
    
    
    # env.front_camera.get_rgb_graph(save_or_not=True, save_path="./front.png")
    
    
    if error_type == "object_blocked":
        extra_env = get_env(subtask, front_img, prompts['get_env_block'])
        replan = call_gpt_llm(
            client=client,
            prompt1=prompts['grasp_replan_external'],
            prompt2=extra_env,
            prompt3=prompts['action'],
            task=following_subtasks,
            error_message=error_message,
        )
    elif error_type == "goal_too_far":
        replan = call_gpt_llm(
            client=client,
            prompt1=prompts['grasp_replan_tool'],
            prompt2=prompts['action'],
            task=following_subtasks,
            error_message=error_message,
    )
    else:
        extra_env = get_env(subtask, front_img, prompts['get_env_support'])
        replan = call_gpt_llm(
            client=client,
            prompt1=prompts['grasp_replan_external'],
            prompt2=extra_env,
            prompt3=prompts['action'],
            task=following_subtasks,
            error_message=error_message,
        )
    
    
    print(replan)

    return replan


def load_prompts(directory):
    prompts = dict()
    
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        
        if os.path.isfile(path) and filename.endswith('.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            key = filename[:-4]  

            prompts[key] = content
    
    return prompts


def filter_contact_points_quadrant(pcd, center, cur_ori, target_ori, min_radius=0.0, eps=1e-9):
    """
    pcd: (N,3) 点云（世界系）
    center: (3,) 物体中心（世界系）
    cur_ori, target_ori: 四元数 (w,x,y,z)
    返回: mask (N,)，True 表示满足“半轴 + 1/4 区域”的候选接触点
    """
    def quat_to_R(q):
        q = np.asarray(q, float); q /= (np.linalg.norm(q)+eps)
        w,x,y,z = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
        ])

    def signed_angle_2d(a, b):
        a = a / (np.linalg.norm(a)+eps)
        b = b / (np.linalg.norm(b)+eps)
        cr = a[0]*b[1] - a[1]*b[0]      # z 分量（a×b）
        dt = a[0]*b[0] + a[1]*b[1]      # a·b
        return np.arctan2(cr, dt)       # (-pi, pi]

    # 当前/目标 X 轴在 XY 平面的单位向量
    Rx3 = quat_to_R(cur_ori)[:, 0]
    Tx3 = quat_to_R(target_ori)[:, 0]
    Rx = Rx3[:2]; Rx /= (np.linalg.norm(Rx)+eps)
    Tx = Tx3[:2]; Tx /= (np.linalg.norm(Tx)+eps)

    # 有符号夹角：从 Rx 旋到 Tx（逆时针>0，顺时针<0）
    theta = signed_angle_2d(Rx, Tx)
    print("theta:", theta)
    abs_theta = abs(theta)
    sign = 1.0 if theta >= 0 else -1.0

    # Rx 的法向（逆时针 90°）
    Rperp = np.array([-Rx[1], Rx[0]])

    # 点相对中心的 XY 向量
    rel_xy = pcd[:, :2] - center[:2]
    rel_dot_Rx = rel_xy @ Rx          # 到 Rx 的投影
    rel_dot_Rp = rel_xy @ Rperp       # 到 Rperp 的投影
    radii = np.linalg.norm(rel_xy, axis=1)

    # 规则 1：半轴（夹角小于 90° 取 Rx 正半轴，否则负半轴）
    half_sign = +1.0 if abs_theta < (np.pi/2) else -1.0
    mask_half = (rel_dot_Rx * half_sign) > 0.0

    # 规则 2：1/4 区域（按旋转方向取 Rx 的左/右侧）
    # theta>0 取左侧: rel·Rperp > 0；theta<0 取右侧: rel·Rperp < 0
    mask_quarter = (rel_dot_Rp * sign) > 0.0

    # 可选：剔除太靠近中心的点，避免数值不稳定或“拖拽”
    mask_radius = radii > (min_radius + eps)

    mask = mask_half & mask_quarter & mask_radius
    return mask, np.degrees(theta) 



def push_to_target(env, target_position, t_diff=0.1, o_diff = 15):
    env.franka.close_gripper()
    print("target_position", target_position)
    ni = 0
    loss_t = 1e9
    loss_o = 1e9
    I = 0.0            # 积分项
    prev_err = 0.0     # 上一步误差
    while (loss_t > t_diff or loss_o > o_diff) and ni < 5:
        ni = ni+1
        print("第", ni, "次调整")
        print("loss_t1", loss_t)
        print("loss_o1", loss_o)
        print("t_diff", t_diff)
        print("o_diff", o_diff)
        if loss_t > t_diff:
            initial_xyz = env.object._prim.get_world_pose()[0]
            print("initial_xyz", initial_xyz)
            target_xyz  = np.array(target_position[0], dtype=float)
            vector = target_xyz - initial_xyz
            vector[2] = 0
            set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
            set_prim_visible_group(prim_path_list=["/World/Subtarget"], visible=False)
            
            for i in range(25):
                env.step()
            pcd, _  = env.top_camera.get_point_cloud_data_from_segment(
                    sample_flag=True,
                    sampled_point_num=4096,
                    real_time_watch=False
            )
            for i in range(5):
                env.step()
            set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
            set_prim_visible_group(prim_path_list=["/World/Subtarget"], visible=True)

            hull = ConvexHull(pcd)
            hull_points = pcd[hull.vertices]   # 只取凸包顶点
            # hull2d = ConvexHull(pcd[:, :2])
            # hull_points = pcd[hull2d.vertices]
            d = np.linalg.norm(vector)
            if d < 1e-9:
                print("The initial position is too close to the target position.")
            u = vector / d
            u_neg = -u
            # 每个点到半直线的参数 t（沿 -u 的投影长度）
            rel = hull_points - initial_xyz                      # (N,3)
            t = rel @ u_neg                              # (N,)

            # 只保留在半直线上、且离 initial 沿线为正的点
            mask = t > 0.0
            if not np.any(mask):
                raise ValueError("No points lie in the opposite half-line region")

            rel_m = rel[mask]                            # (M,3)
            t_m = t[mask][:, None]                       # (M,1)

            # 线到点的向量（去掉沿线分量），即垂直残差
            perp_vec = rel_m - t_m * u_neg               # (M,3)
            perp_dist = np.linalg.norm(perp_vec, axis=1) # (M,)

            best_idx = np.argmin(perp_dist)
            contact_point = hull_points[mask][best_idx]
            debug_tool1 = VisualCuboid(
                    prim_path="/World/Debug_cube",
                    name="debug_cube",
                    scale=np.array([0.01, 0.01, 0.01]),
                    color=np.array([1.0, 0.0, 0.0]),
                    translation=np.array(contact_point),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    visible=False,
                )
                # 刷新几步渲染才能看到
            for _ in range(5):
                env.step()
            offset = 0.03
            v_offset = offset * u_neg
            print("v_offset", v_offset)
            print("contact_point", contact_point)
            initial_position = contact_point + v_offset
            # print("Initial xzy:", initial_xyz)
            print("Moving to initial position:", initial_position) 
            pre_initial_position = initial_position.copy()
            pre_initial_position[2] += 0.2
            env.franka.Rmpflow_Move(
                target_position=np.array(pre_initial_position),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            print("pre_initial_position", pre_initial_position)
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(initial_position),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(initial_position),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            # offset2 = 0.03
            # v_offset2 = offset2 * u_neg
            # target = initial_position+vector-v_offset
            target = initial_position+vector- 0.8*v_offset
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(target),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            if t_diff < 0.01:
                env.franka.Rmpflow_Move(
                    target_position=np.array(target),
                    target_orientation=np.array([180.0, 0.0, 0.0]),
                )
            
            middle_point = target.copy()
            middle_point[2] += 0.2

            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(middle_point),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            # env.franka.Rmpflow_Move(
            #     target_position=np.array([0.2, 0, 1.0]),
            #     target_orientation=np.array([180.0, 0.0, 0.0]),
            # )

            for _ in range(15):
                env.step()
            
            if env.object._prim.get_world_pose()[0][2] < 0.75:
                return False
            
            loss_t = np.linalg.norm(env.object._prim.get_world_pose()[0]- target_xyz)

        ### align the yaw angle, first get the operation point and then get the direaction
        print('target_ori', target_position[1])
        print('cur_ori', env.object._prim.get_world_pose()[1])
        object_center = env.object._prim.get_world_pose()[0]
        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
        set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False)
        
        for _ in range(15):
            env.step()

        if object_center[2] < 0.75:
            return False

        for i in range(25):
            env.step()
        pcd, _  = env.top_camera.get_point_cloud_data_from_segment(
                sample_flag=True,
                sampled_point_num=4096,
                real_time_watch=False
        )
        for i in range(5):
            env.step()
        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
        set_prim_visible_group(prim_path_list=["/World/Subtarget"], visible=True)
        hull = ConvexHull(pcd)
        hull_points = pcd[hull.vertices]   # 只取凸包顶点
        # hull2d = ConvexHull(pcd[:, :2])
        # hull_points = pcd[hull2d.vertices]
        mask, theta = filter_contact_points_quadrant(hull_points, object_center, target_position[1],env.object._prim.get_world_pose()[1])
        loss_o = abs(theta)
        if loss_o > o_diff:
            print("Current loss in orientation:", loss_o)
            candidate_points = hull_points[mask]
            print("Candidate points shape:", candidate_points.shape)
            if candidate_points.shape[0] > 0:
                dists = np.linalg.norm(candidate_points[:, :2] - object_center[:2], axis=1)
                best_idx = np.argmax(dists)
                contact_point = candidate_points[best_idx]
                debug_tool = VisualCuboid(
                    prim_path=f"/World/Debug/Cube",   # 每个点不同的路径
                    name=f"debug_cube",
                    scale=np.array([0.01, 0.01, 0.01]),   # 小一点避免遮挡
                    color=np.array([0.0, 1.0, 0.0]),      # 绿色
                    translation=np.array(contact_point),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    visible=False,
                )
                for _ in range(10):
                    env.step()
            r_vector = contact_point[:2] - object_center[:2]
            r_vector /= np.linalg.norm(r_vector) + 1e-9
            if theta < 0:
                perpe = np.array([-r_vector[1], r_vector[0]])   # CCW
            else:
                perpe = np.array([ r_vector[1],-r_vector[0]])   # CW
            perpe = perpe / (np.linalg.norm(perpe) + 1e-9)
            # I = np.clip(I+theta, -10, 10)  # I is limited in 0.1
            # print("I", I)
            Kp = 0.003
            Kd = 0.001
            de = loss_o - prev_err
            u = Kp * loss_o + Kd * de  # PI controller
            u = np.clip(u, 0.0, 0.1) 
            print("Control input u:", u)
            prev_err = loss_o
            pre_offset = 0.05
            robot_xy_target = contact_point[:2] + u * perpe 
            robot_xy_start = contact_point[:2] - pre_offset * perpe
            initial_pos = [robot_xy_start[0], robot_xy_start[1], contact_point[2]]
            target_pos = [robot_xy_target[0], robot_xy_target[1], contact_point[2]]
            print("Initial position:", initial_pos)
            print("Target position:", target_pos)
            pre_initial_pos = initial_pos.copy()
            pre_initial_pos[2] += 0.3
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(pre_initial_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(pre_initial_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(initial_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(initial_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(target_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(target_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )

            middle = target_pos.copy()
            middle[2] += 0.5
            # middle[0] += 0.2
            
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(middle),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )

            # env.franka.Rmpflow_Move(
            #     target_position=np.array([0.2, 0, 1.0]),
            #     target_orientation=np.array([180.0, 0.0, 0.0]),
            # )

            for _ in range(15):
                env.step()

            if env.object._prim.get_world_pose()[0][2] < 0.75:
                return False
        
        print("loss_o", loss_o)
        print("loss_t", loss_t)

    if(loss_t > t_diff or loss_o > o_diff):
        print("Failed to push to target position.")
        return False
    else:
        print("Successfully pushed to target position!")
        return True

def move_to_target(env, target_position):
    """
    根据物体目标位姿计算并执行机械臂目标位姿。
    - target_position: 目标物体位置和四元数 [位置, 四元数]。
    """
    # 获取当前末端执行器位置和物体位置
    ee_pos, ee_quat = env.franka.get_cur_ee_pos()  # 机械臂末端执行器的当前位姿（位置和旋转）
    obj_pos, obj_quat = env.object._prim.get_world_pose()  # 物体当前位姿（位置和旋转）

    obj_rotation = quat_to_euler_angles(obj_quat, degrees=True)

    # 目标位置和目标旋转四元数
    target_pos = np.array(target_position[0], dtype=float)
    target_quat = np.array(target_position[1], dtype=float)

    target_rotation = quat_to_euler_angles(target_quat, degrees=True)

    print("obj_rotation", obj_rotation)
    print("target_rotation", target_rotation)

    # 计算当前末端执行器与物体中心的相对位姿（假设刚性抓取）
    vector = ee_pos - obj_pos  # 末端执行器和物体之间的相对位置向量
    print("vector", vector)
    print("target_pos", target_pos)

    # 计算目标物体目标位置的相对位置（末端相对物体的位姿保持不变）
    new_gripper_pos = target_pos + vector  # 新的末端执行器位置

    # 先调整 x, y 方向
    new_gripper_pos[2] = 1.0  # 保持z轴为1.0
    print("new_gripper_pos", new_gripper_pos)


    # 调整机械臂末端执行器位置（移动到新的 x, y 位置）
    env.franka.Rmpflow_Move(
        target_position=np.array(new_gripper_pos),
        target_orientation=np.array(ee_quat),
        quat_or_not=True
    )
    env.franka.Dense_Rmpflow_Move(
        target_position=np.array(new_gripper_pos),
        target_orientation=np.array(ee_quat),
        quat_or_not=True
    )
    rotation = quat_to_euler_angles(ee_quat, degrees=True)

    if(abs(obj_rotation[0]-target_rotation[0])>75):
        rotation[0] = rotation[0] - 30

    env.franka.Rmpflow_Move(
        target_position=np.array(new_gripper_pos),
        target_orientation=np.array(rotation),
        quat_or_not=False
    )

    # 然后调整z轴，使末端执行器z轴与目标位置一致
    new_gripper_pos[2] = 0.85  # 将末端执行器z轴调整为目标物体z轴的位置

    # 最后移动到目标位置
    env.franka.Dense_Rmpflow_Move(
        target_position=np.array(new_gripper_pos),
        target_orientation=np.array(rotation),
        quat_or_not=False
    )

    ee_pos, ee_quat = env.franka.get_cur_ee_pos()

    print("Move to target position completed:", ee_pos)
    return True
    
def release(env, t_pos):
    env.franka.open_gripper()
    ee_pos, ee_quat = env.franka.get_cur_ee_pos()  # 机械臂末端执行器的当前位姿（位置和旋转）
    obj_pos, obj_quat = env.object._prim.get_world_pose()  # 物体当前位姿（位置和旋转）

    vector = ee_pos - obj_pos  # 末端执行器和物体之间的相对位置向量
    release_vector = vector / np.linalg.norm(vector) * 0.2
    new_ee_pos = ee_pos + release_vector

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array(new_ee_pos),
        target_orientation=ee_quat,  # 继续保持末端执行器当前的姿态
        quat_or_not=True
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([new_ee_pos[0], new_ee_pos[1], 1.0]),
        target_orientation=np.array([180.0, 0.0, 0.0]),
        quat_or_not=False
    )

    # ik = check_pose(env, t_pos, pos_thresh=0.2, angle_thresh=400.0)
    # return ik
    return True


def rotate_up_target(env,t_pos):
    env.franka.close_gripper()
    obj_pos, obj_quat = env.object._prim.get_world_pose()  # 物体当前位姿（位置和旋转）

    #  ##push的初始位置
    # env.franka.Rmpflow_Move(
    #     target_position=np.array([0.2, obj_pos[1], 0.765]),
    #     target_orientation=np.array([180.0, 0.0, 0.0]),
    # )
    
    # ##push到墙壁
    # env.franka.Dense_Rmpflow_Move(
    #     target_position=np.array([0.14, obj_pos[1], 0.765]),
    #     target_orientation=np.array([180.0, 0.0, 0.0]),
    # )

    # env.franka.Dense_Rmpflow_Move(
    #     target_position=np.array([0.18, obj_pos[1], 0.8]),
    #     target_orientation=np.array([180.0, 0.0, 0.0]),
    # )

    

    obj_pos, obj_quat = env.object._prim.get_world_pose()
    
    contact_point = obj_pos
    print(obj_pos)
    contact_point[0] = obj_pos[0] + 0.06
    contact_point[2] = 0.74
    print(contact_point)

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([0.2, 0.0, 0.765]),
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=contact_point,
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    contact_point[0] = contact_point[0] - 0.01
    contact_point[2] = 0.77

    env.franka.Rmpflow_Move(
        target_position=contact_point,
        target_orientation=np.array([180.0, 15.0, 0.0]),
    )

    contact_point[0] = contact_point[0] + 0.02
    contact_point[2] = contact_point[2] - 0.05
    env.franka.Dense_Rmpflow_Move(
        target_position=contact_point,
        target_orientation=np.array([180.0, 37.0, 0.0]),
    )
    for i in range(20):
        env.step()
    
    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([0.2, 0, 1.0]),
        target_orientation=np.array([180.0, 0.0, 0.0]),
        quat_or_not=False
    )

    ik = check_pose(env, t_pos, pos_thresh=1.0, angle_thresh=15.0)
    return ik
    

def rotate_down_target(env, t_pos):
    env.franka.close_gripper()
    def contactPoint(env):
        obj_pos = env.object._prim.get_world_pose()[0]
        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
        set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False) 
        for _ in range(25):
            env.step()

        pcd, _  = env.top_camera.get_point_cloud_data_from_segment(
            sample_flag=True,
            sampled_point_num=4096,
            real_time_watch=True
        )

        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
        set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=True)

        # —— 凸包 & 最左点（x最小）
        hull = ConvexHull(pcd)
        hull_points = pcd[hull.vertices]                      # (K,3)

        min_idx = np.argmin(hull_points[:, 0])
        cp = np.array(hull_points[min_idx], dtype=float).copy()  # 强制拷贝为独立 ndarray

        # 用物体中心的 y 覆盖（如果你需要这样做）
        cp[1] = float(obj_pos[1])
        cp[2] = float(np.max(hull_points[:, 2]))

        return cp  # 这里返回的是独立拷贝

    # 1) 独立的 contact_point
    contact_point = np.array(contactPoint(env), dtype=float).reshape(3,).copy()

    # 2) 先得到真正的“起点” (按你的偏移规则)
    startpoint = contact_point.copy()
    startpoint[0] -= 0.03
    # startpoint[2] -= 0.05

    # 3) 再基于 startpoint 生成 pre_point / end_point（注意每一步都 copy）
    pre_point = startpoint.copy()
    pre_point[2] += 0.05

    end_point = startpoint.copy()
    end_point[0] += 0.35

    print("contact_point", contact_point)
    print("startpoint", startpoint)
    print("pre_point", pre_point)
    print("end_point", end_point)

    # 如果这两个API接受 np.ndarray 就直接传；否则 .tolist()
    env.franka.Dense_Rmpflow_Move(
        target_position=pre_point,
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    env.franka.Rmpflow_Move(
        target_position=startpoint,
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=end_point,
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    ik = check_pose(env, t_pos, pos_thresh=1.0, angle_thresh=20.0)
    return ik



    


def check_pose(env, t_pos, pos_thresh=0.15, angle_thresh=20.0):
    obj_pos = env.object._prim.get_world_pose()
    # --- 提取 ---
    t_xyz, t_quat = np.array(t_pos[0]), np.array(t_pos[1])
    o_xyz, o_quat = np.array(obj_pos[0]), np.array(obj_pos[1])

    # --- 位置差 ---
    pos_diff = np.linalg.norm(t_xyz - o_xyz)

    # --- 四元数差 ---
    t_quat = t_quat / np.linalg.norm(t_quat)
    o_quat = o_quat / np.linalg.norm(o_quat)
    dot = abs(np.dot(t_quat, o_quat))
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2 * np.arccos(dot)
    angle_deg = np.degrees(angle_rad)

    print(f"位置差: {pos_diff:.4f} m, 角度差: {angle_deg:.2f}°")
    # --- 判断 ---
    is_close = (pos_diff < pos_thresh) and (angle_deg < angle_thresh)
    
    return is_close


def generate_pose(env, prompts, subtasks, idx, obs_image, subtask, action):
    init_pos = env.object._prim.get_world_pose()
    if action == 'push' or action == 'rotate down' or action == 'rotate up':
        obj = subtask["parameters"]["object"]
        place = subtask["parameters"]["place"]
        task = f"Output the points (<point>x y</point>) for 'the {place}, note that the point should be as close to the {obj} as possible'"
    elif action == 'move to' :
        place = subtask["parameters"]["place"]
        task = f"Output the points (<point>x y</point>) for 'the {place}'"
    print("Task:", task)
    set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False)
    for _ in range(20):
        env.step()
    env.top_camera.get_rgb_graph(save_or_not=True, save_path = "./topdown.png")
    set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=True)
    point_2d=get_2d_point(obs_image, task)
    x, y = point_2d[0]
    point_3d=pixels_to_world(env, x, y, env.top_camera.get_depth_graph()) 

    if point_3d[0] > 0.3 or point_3d[0] < -0.3 or point_3d[1] > 0.3 or point_3d[1] < -0.3:
        print("The point is out of range.")
        return False, None, None
    
    if "table edge" in place:
        if point_3d[1] > -0.18 or point_3d[1] < 0.18:
            print("not near the table edge")
            return False, None, None
    
    if action == 'push':
        image_paths, valid_pos = pose_push_gen(point_3d[0], point_3d[1], env)
    elif action == 'rotate up':
        image_paths, valid_pos = pose_rotate_gen(point_3d[0], point_3d[1], env)
    elif action == 'rotate down':
        image_paths, valid_pos = pose_rotate_down_gen(point_3d[0], point_3d[1], env)
    elif action == 'move to':
        image_paths, valid_pos = pose_move_gen(point_3d[0], point_3d[1], env)

    current_subtask = subtask
    next_subtask = subtasks[idx + 1]
    pose_subtasks = [current_subtask, next_subtask]
    if image_paths is None or len(image_paths) == 0:
        return False, None, None
    num = choose_pose(pose_subtasks, image_paths, prompts['choose_pose'])
    if num < 0 or num >= len(valid_pos):
        num = len(valid_pos) 
        return False, None, None
    target_pos = valid_pos[num]

    env.object._prim.set_world_pose(init_pos[0], init_pos[1])
    for _ in range(20):
        env.step()
    
    return action, init_pos, target_pos

def check_grasp(env, front=False):
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
    
    for i in range(25):
        env.step()
    
    
    if front:
        # env.top_front_camera.get_rgb_graph(save_or_not=True, save_path="./camera.png")
        pc_seg, color_seg = env.top_front_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            save_path="./pc_seg.ply",
            sample_flag=False,
            real_time_watch=False
        )

        pc_scene, color_scene = env.top_front_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            sample_flag=False,
            # workspace_x_limit=[-0.55, 0.55],
            workspace_z_limit=[0.50, None],
        )
    else:
        # env.gripper_camera.get_rgb_graph(save_or_not=True, save_path="./camera.png")
        pc_seg, color_seg = env.gripper_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            save_path="./pc_seg.ply",
            sample_flag=False,
            real_time_watch=False
        )
        
        pc_scene, color_scene = env.gripper_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            sample_flag=False,
            # workspace_x_limit=[-0.55, 0.55],
            workspace_z_limit=[0.50, None],
        )
    
    grasp_check, grasp_position, grasp_orientation = grasp_checker(env, env.top_front_camera, pc_seg, pc_scene, color_scene, vis_grasp_flag=True)

    return grasp_check, grasp_position, grasp_orientation



def execute_grasp(env, pos):
    # env.top_front_camera.get_rgb_graph(save_or_not=True, save_path="images/top_front.png")
    env.franka.open_gripper()
    front = False
    if(pos[1] < -0.18):
        front = True
    if(pos[2]>0.85):
        front = True
    
    grasp_check, grasp_position, grasp_orientation  = check_grasp(env, front)
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
    for i in range(25):
        env.step()
    print(grasp_check)
    if not grasp_check:
        return False, False
    

    x = env.object._prim.get_world_pose()[0][0]
    y = env.object._prim.get_world_pose()[0][1]
    z = env.object._prim.get_world_pose()[0][2]

    if(x>grasp_position[0]):
        pre_x = grasp_position[0] - 0.025
        grasp_position[0] = grasp_position[0] + 0.01
    else:
        pre_x = grasp_position[0] + 0.025
        grasp_position[0] = grasp_position[0] - 0.01
    
    if(y>grasp_position[1]):
        pre_y = grasp_position[1] - 0.025
        grasp_position[1] = grasp_position[1] + 0.01
    else:
        pre_y = grasp_position[1] + 0.025
        grasp_position[1] = grasp_position[1] - 0.01
    
    if(z>grasp_position[2]):
        pre_z = grasp_position[2] - 0.05
        grasp_position[2] = grasp_position[2] + 0.015
    else:
        pre_z = grasp_position[2] + 0.05
        grasp_position[2] = grasp_position[2] - 0.015



    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([0.2, 0, 1.0]),
        target_orientation=np.array([180.0, 0.0, 0.0]),
        quat_or_not=False
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([pre_x, 0, 1.0]),
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    for _ in range(20): 
        env.step()

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([pre_x, pre_y, 1.0]),
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([pre_x, pre_y, pre_z]),
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    for _ in range(20): 
        env.step()

    env.franka.Dense_Rmpflow_Move(
        target_position=grasp_position,
        target_orientation=grasp_orientation,
        quat_or_not=True
    )
    
    env.franka.Rmpflow_Move(
        target_position=grasp_position,
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    for _ in range(20): 
        env.step()
    
    env.franka.close_gripper()

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([grasp_position[0], grasp_position[1], 1.0]),
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    if(env.object._prim.get_world_pose()[0][2] > z+0.1):
        return True, True
    else:
        return False, False


def NP(env, task_instruction):

    env.top_camera.get_rgb_graph(save_or_not=True, save_path = "./topdown.png")
    obs_image = "./topdown.png"
    env.front_camera.get_rgb_graph(save_or_not=True, save_path="./front.png")
    front_image = "./front.png"
    # obs_image, depth = get_image_depth()
    # front_image = obs_image.copy()
    prompt_directory = '/home/zjx/Downloads/LLM_TAMP/vlm_plan/prompts'
    prompts = load_prompts(prompt_directory)
    first_plan = plan(task_instruction, obs_image, prompts['firstplan'], prompts['action'])
    subtasks = split_actions(first_plan)
    obj = subtasks[0]["parameters"]["object"]
    # print("subtasks", subtasks)
    # print("The object to be manipulated is:", obj)
    save_path = "./np1.mp4"
    breakpoint()
    for idx, subtask in enumerate(subtasks):
        temp = 0
        obj_o = env.object._prim.get_world_pose()
        is_last = False
        while temp < 4 :
            if subtask['action'] == 'grasp':
                pose, ik = execute_grasp(env, obj_o[0])
                print("[INFO] Grasp Pose:", pose)
                print("[INFO] IK Solution:", ik)
            else:
                pose = True
                if  idx == len(subtasks) - 1 or (idx == len(subtasks) - 2 and subtasks[idx+1]['action'] == 'release'):
                    act = subtask['action']
                    i_pos = env.object._prim.get_world_pose()
                    t_pos = env.non_collision_object._prim.get_world_pose()
                    if "push" in act:
                        temp = 4
                else:
                    act, i_pos, t_pos = generate_pose(env, prompts, subtasks, idx, obs_image, subtask, subtask['action'])

                if not act:
                    ik = False
                elif "move to" in act:
                    ik = move_to_target(env, t_pos)
                elif "release" in act:
                    ik = release(env,t_pos)
                elif "push" in act:
                    ik = push_to_target(env, t_pos)
                elif "rotate up" in act:
                    ik = rotate_up_target(env, t_pos)
                elif "rotate down" in act:
                    ik = rotate_down_target(env, t_pos)


                print("act", act)
                
                # if not act:
                #     ik = False
                # elif subtask['action'] == 'move to' or subtask['action'] == 'release':
                #     ik = True
                # else:
                #     ik = check_pose(env, t_pos)
            
            if ik and pose:
                break
            else:
                if subtask['action'] != 'move to' or subtask['action'] != 'release':
                    env.object._prim.set_world_pose(obj_o[0], obj_o[1])
                    temp = temp + 1      
      
        # if idx == len(subtasks) - 1:
        #     obj_o = env.object._prim.get_world_pose()
        #     if not check_pose(env, t_pos, 0.05, 5):
        #         while env.object._prim.get_world_pose()[0][2] > t_pos[0][2]+0.05:
        #             env.object._prim.set_world_pose(obj_o[0], obj_o[1])
        #             rotate_down_target(env, t_pos)
        #         ik = push_to_target(env, t_pos, 0.007, 5)
        #         if not ik:
        #             subtasks[idx+1:] = [
        #                 {"action": "push","parameters": {"object": obj, "place": "transparent target"}}
        #             ]
        #             idx = idx + 1
                
                
        if not pose or not ik:
            following_subtasks = subtasks[idx:]
            new_plan = replan(env, subtask, following_subtasks, obs_image, front_image, prompts, pose, ik)
            replan_subtasks = split_actions(new_plan)
            subtasks[idx+1:] = replan_subtasks
    
    if not check_pose(env, t_pos, 0.05, 5):
        if env.object._prim.get_world_pose()[0][2] > t_pos[0][2]:
            rotate_down_target(env,t_pos)
        
        push_to_target(env, t_pos, 0.01, 5)
    
    print("successfully finish the task!")
    env.camera.create_mp4(save_path)

def align_real_sim(env):
    real_pose = '/home/zjx/Downloads/LLM_TAMP/realworld/real_pose.npy'
    pose = np.load(real_pose)
    position = pose[0, :3]
    rotation = pose[0, 3:]
    env.object._prim.set_world_pose(position, rotation)
    for i in range(30):
        env.step()

def real_world_gen(env):
    align_real_sim(env)
    prompt_directory = '/home/zjx/Downloads/LLM_TAMP/vlm_plan/prompts'
    prompts = load_prompts(prompt_directory)
    breakpoint()
    target_center = '/home/zjx/Downloads/LLM_TAMP/realworld/sim_target_center.npy'
    target_center = np.load(target_center)
    print("Loaded center:", target_center)

    image_paths, valid_pos = pose_push_gen(target_center[0], target_center[1], env)
    plan1 = plan()
    subtasks = split_actions(plan1)
    current_subtask = subtasks[0]
    next_subtask = subtasks[1]
    pose_subtasks = [current_subtask, next_subtask]
    print(pose_subtasks)
    breakpoint()
    # TODO: finish the generation
    if image_paths is None or len(image_paths) == 0:
        return False, None, None
    pre_path = "/home/zjx/Downloads/LLM_TAMP/"
    # 循环处理
    for i in range(len(image_paths)):
        # 去掉前面的 './' 再拼接
        image_paths[i] = pre_path + image_paths[i].lstrip("./")
    print(image_paths)
    num = choose_pose(pose_subtasks, image_paths, prompts['choose_pose'])
    if num < 0 or num >= len(valid_pos):
        num = len(valid_pos) 
        return False, None, None
    target_pos = valid_pos[num]
    print(target_pos)

    return target_pos


def main():
    # sim_pt = np.load("sim_pt.npy")
    # print("Loaded sim_pt:", sim_pt)
    # # breakpoint()

    # task_instruction = "Move the blue book to the transparent target."
    env = Demo_Scene_Env()
    # env.thread_record.start()
    # for _ in range(20):
    #     env.step()
    
    # env.gripper_camera.get_rgb_graph(save_or_not=True, save_path = "./camera.png")
    
    # print(env.object._prim.get_world_pose()[0])
    # NP(env, task_instruction)
    target_pose = real_world_gen(env)
    breakpoint()
    # execute_grasp(env, env.object._prim.get_world_pose()[0][1])
    # t_pos = env.non_collision_object._prim.get_world_pose()
    # move_to_target(env, t_pos)
    # release(env)
    # rotate_down_target(env, t_pos)
    # push_to_target(env, t_pos, 0.1, 15)
    # push_to_target(env, t_pos, 0.01, 5)



    

if __name__ == "__main__":
    main()
    while simulation_app.is_running():
        simulation_app.update()
    
simulation_app.close()