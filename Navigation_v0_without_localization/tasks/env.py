import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import carb
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import clip
import torchvision.transforms as T
from typing import Optional
from scipy.special import expit
from pprint import pprint 
from dataclasses import asdict, dataclass
from configs.main_config import MainConfig

import psutil
from ultralytics import YOLO
import cv2

from .control_manager import PID_controller, Control_module
from .scene_manager import Scene_controller
from .memory_manager import ImageMemoryManager
from .graph_manager import Graph_manager
import random

sim_config = {
    "renderer": "RayTracedLighting",
    "headless": True,
    #"headless": False,
    "multi_gpu": False, 
    #"active_gpu": gpu_to_use,
    "enable":"omni.kit.livestream.native"
}

GET_DIR = False

def euler_from_quaternion(vec):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = vec[0], vec[1], vec[2], vec[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def get_quaternion_from_euler(roll,yaw=0, pitch=0):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.array([qx, qy, qz, qw])

import time
from embed_nn import SceneEmbeddingNetwork
import torch.optim as optim
from collections import deque
import os

# В начале файла добавим импорт
from .graph_data_collector import GraphDataCollector

class CLGRCENV(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config = MainConfig(),
        skip_frame=4,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=2048,
        seed=10,
        MAX_SR=50,
        test=False,
        reward_mode=0
    ) -> None:
        self.init = True
        from omni.isaac.kit import SimulationApp
        self.config = config
        sim_config["headless"] = asdict(config).get('headless', None)
        self._simulation_app = SimulationApp(sim_config)
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from .wheeled_robot import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.core.objects import VisualCuboid, FixedCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim
        from pxr import Semantics
        self.scene_controller = Scene_controller()
        self.controle_module = Control_module()
        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()
        # Для пола (в __init__)
        import omni.usd
        self.my_stage = omni.usd.get_context().get_stage()
        ground_prim = self.my_stage.GetPrimAtPath("/World/defaultGroundPlane")
        semantics_api = Semantics.SemanticsAPI.Apply(ground_prim, "Semantics")
        semantics_api.CreateSemanticTypeAttr().Set("class")
        semantics_api.CreateSemanticDataAttr().Set("floor")
        print(f"Semantic tag for /World/defaultGroundPlane: {semantics_api.GetSemanticDataAttr().Get()}")
        # ground_prim = self.my_stage.GetPrimAtPath("/World/defaultGroundPlane")
        # ground_prim.SetCustomDataByKey("semantic:tag", "floor")
        # Словарь для отслеживания созданных конфигураций
        self.scene_configs = {}  # {scene_id: config_dict}
        self.current_scene_id = None

        # Базовая директория для вывода
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_data")

        # Остальная инициализация (без робота пока)
        room_usd_path = asdict(config).get('room_usd_path', None)
        create_prim(prim_path="/room", translation=(0, 0, 0), usd_path=room_usd_path)
        jetbot_asset_path = asdict(config).get('jetbot_asset_path', None)
        self.jetbot = self._my_world.scene.add(
            WheeledRobot(
                prim_path="/jetbot",
                name="my_jetbot",
                wheel_dof_names=["left_wheel", "right_wheel"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([-10, -10, 0.0]),
                orientation=get_quaternion_from_euler(-np.pi/2),
            )
        )
        # Отложим создание робота до reset
        # self.jetbot = None
        # self.jetbot_controller = None

        # Остальная часть __init__ остаётся без изменений
        from pxr import PhysicsSchemaTools, UsdUtils, PhysxSchema, UsdPhysics
        from pxr import Usd
        from omni.physx import get_physx_simulation_interface
        self.my_stage = omni.usd.get_context().get_stage()
        self.my_prim = self.my_stage.GetPrimAtPath("/jetbot")

        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.my_prim)
        contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)
        # self.set_env(config)
        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)

        self.goal_position = np.array([0,0,0])
        self.render_products = []
        from omni.replicator.isaac.scripts.writers.pytorch_writer import PytorchWriter
        from omni.replicator.isaac.scripts.writers.pytorch_listener import PytorchListener
        import omni.replicator.core as rep
        self.image_resolution = 400
        self.camera_width = self.image_resolution
        self.camera_height = self.image_resolution

        camera_path_1 = "/jetbot/fl_link4/visuals/realsense/husky_rear_left"#asdict(config).get('camera_paths', None)
        camera_path_2 = "/jetbot/fr_link4/visuals/realsense/husky_rear_left"#asdict(config).get('camera_paths', None)
        render_product = rep.create.render_product(camera_path_1, resolution=(self.camera_width, self.camera_height))
        self.render_products.append(render_product)
        render_product = rep.create.render_product(camera_path_2, resolution=(self.camera_width, self.camera_height))
        self.render_products.append(render_product)
        self.key = [0,0]

        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = PytorchListener()
        self.pytorch_writer = rep.WriterRegistry.get("PytorchWriter")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#"cpu")
        print("device = ", self.device)
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device=self.device)
        self.pytorch_writer.attach(self.render_products)

        self.seed(seed)
        self.reward_range = (-10000, 10000)
        
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        shape_size = 1542
        self.use_memory = asdict(config).get('memory', None)
        print("use memory", self.use_memory)
        if self.use_memory:
            self.memory = ImageMemoryManager()
        if self.use_memory:
            shape_size += 2048

        self.use_graph = asdict(config).get('graph', None)
        if self.use_graph:
            self.graph_module = Graph_manager()
        if self.use_graph:
            shape_size += 5698

        self.observation_space = spaces.Box(low=-1000000000, high=1000000000, shape=(1542,), dtype=np.float32) #2060

        self.max_velocity = 1.5
        self.max_angular_velocity = math.pi*0.4
        self.events = [1]#[0, 1, 2]
        self.event = 0

        convert_tensor = transforms.ToTensor()

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

        goal_path = asdict(config).get('goal_image_path', None)

        img_goal = clip_preprocess(Image.open(goal_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.img_goal_emb = self.clip_model.encode_image(img_goal)
            self.start_emb = self.img_goal_emb

        self.model = YOLO("yolov8m-seg.pt")
        #self.model.to('cuda')
        self.stept = 0
        
        self.collision = False
        self.start_step = True
        self.MAX_SR = MAX_SR
        self.num_of_step = 0
        self.steps_array = []
        self.reward_modes = ["move", "rotation"]#, "Nan"]
        self.reward_mode = asdict(config).get('reward_mode', None)
        self.local_reward_mode = 0
        self.delay_change_RM = 0
        self.prev_SR = {}
        self.log_path = asdict(config).get('log_path', None)

        self.training_mode = asdict(config).get('training_mode', None)
        self.local_training_mode = 0
        self.traning_radius = 0
        self.trining_delta_angle = 0
        self.max_traning_radius = 4
        self.max_trining_angle = np.pi/3
        self.amount_angle_change = 0
        self.amount_radius_change = 0
        self.max_amount_angle_change = 3
        self.max_amount_radius_change = 10
        self.num_of_envs = 0
        self.eval = asdict(config).get('eval', None)
        torch.save(torch.tensor([0]), asdict(config).get('loss_path', None))
        self.t = 0
        import omni.isaac.core.utils.prims as prim_utils
        self.learn_emb = 0
        self.current_jetbot_position = np.array([0,0])
        self.current_jetbot_orientation = 0

        self.evalp = asdict(config).get('eval_print', None)
        self.eval_log_path = asdict(config).get('eval_log_path', None)
        self.eval_r = 0.2
        self.eval_angle = 0
        self.eval_dt = 0.2
        self.eval_dangle = np.pi/18
        self.eval_step = 0
        self.eval_step_angle = 0
        self.eval_sr = []
        self.eval_write = 0
        self.SR_len = 10
        self.SR_t = 0
        self.demonstrate = False
        self.get_target_positions = []
        self.level = 0
        self.cache = None

        if self.use_graph:
            self.graph_embedding = self.graph_module.get_graph_embedding(self.num_of_envs)
        self.imitation_part = 1
        self.init = False
        return
         
    
    def get_success_rate(self, observation, terminated, sources, source="Nan"):
        #cut_observation = list(observation.items())[0:6]
        self._insert_step(self.steps_array, self.num_of_step, self.event, observation, terminated, source)
        pprint(self.steps_array)
        print("summary")
        pprint(self._calculate_SR(self.steps_array, self.events, sources))
        
    def _insert_step(self, steps_array, i, event, observation, terminated, source):
         steps_array.append({
            "i": i,
            "event": event,
            "terminated": terminated,
            "source": source,
            "observation": observation,
            })
         if len(steps_array) > self.MAX_SR:
            steps_array.pop(0)

    def _calculate_SR(self, steps_array, events, sources):
        SR = 0
        SR_distribution = dict.fromkeys(events,0)
        step_distribution = dict.fromkeys(events,0)
        FR_distribution = dict.fromkeys(sources, 0)
        FR_len = 0
        for step in steps_array:
            step_distribution[step["event"]] += 1
            if step["terminated"] is True:
                SR += 1
                SR_distribution[step["event"]] += 1
            else:
                FR_distribution[step["source"]] += 1
                FR_len += 1

        for source in sources:
            if FR_len > 0:
                FR_distribution[source] = FR_distribution[source]/FR_len
        for event in events:
            if step_distribution[event] > 0:
                SR_distribution[event] = SR_distribution[event]/step_distribution[event]

        SR = SR/len(steps_array)
        self.prev_SR = SR_distribution
        return  SR, SR_distribution, FR_distribution
    
    def _get_dt(self):
        return self._dt

    def _is_collision(self):
        if self.collision:
            print("collision error!")
            self.collision = False
            return True 
        return False

    def _get_current_time(self):
        return self._my_world.current_time_step_index - self._steps_after_reset

    def _is_timeout(self):
        # print("time: ", self._get_current_time())
        if self._get_current_time() >= self._max_episode_length:
            print("time out")
            return True
        return False

    def get_quadrant(self, nx, ny, vector):
        LR = vector[0]*nx[1] - vector[1]*nx[0]
        mult = 1
        if LR < 0:
            mult = -1
        return mult

    def get_gt_observations(self, previous_jetbot_position, previous_jetbot_orientation):
        goal_world_position = self.goal_position
        current_jetbot_position, current_jetbot_orientation = self.jetbot.get_world_pose()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        entrance_world_position = np.array([0.0, 0.0])

        if self.event == 0:
            dif = 0.9
            entrance_world_position[0] = goal_world_position[0] - dif
            entrance_world_position[1] = goal_world_position[1] - dif
        elif self.event == 1:
            entrance_world_position[0] = goal_world_position[0] + 1
            entrance_world_position[1] = goal_world_position[1]
        else:
            entrance_world_position[0] = goal_world_position[0]
            entrance_world_position[1] = goal_world_position[1] - 1
        goal_world_position[2] = 0

        current_dist_to_goal = np.linalg.norm(goal_world_position[0:2] - current_jetbot_position[0:2])
        self.current_jetbot_position = current_jetbot_position[0:2]
        self.current_jetbot_orientation = euler_from_quaternion(current_jetbot_orientation)[0]
        nx = np.array([-1,0])
        ny = np.array([0,1])
        to_goal_vec = (goal_world_position - current_jetbot_position)[0:2]
        quadrant = self.get_quadrant(nx, ny, to_goal_vec)
        cos_angle = np.dot(to_goal_vec, nx) / np.linalg.norm(to_goal_vec) / np.linalg.norm(nx)
        delta_angle = math.degrees(abs(euler_from_quaternion(current_jetbot_orientation)[0] - quadrant*np.arccos(cos_angle)))
        orientation_error = delta_angle if delta_angle < 180 else 360 - delta_angle

        observation = {
            "entrance_world_position": entrance_world_position, 
            "goal_world_position": goal_world_position, 
            "current_jetbot_position": current_jetbot_position, 
            "current_jetbot_orientation":math.degrees(euler_from_quaternion(current_jetbot_orientation)[0]),
            "jetbot_to_goal_orientation":math.degrees(quadrant*np.arccos(cos_angle)),
            "jetbot_linear_velocity": jetbot_linear_velocity,
            "jetbot_angular_velocity": jetbot_angular_velocity,
            "delta_angle": delta_angle,
            "current_dist_to_goal": current_dist_to_goal,
            "orientation_error": orientation_error,
        }
        return observation

    def change_reward_mode(self):
        if self.start_step:
            self.start_step = False
            if self.delay_change_RM < self.MAX_SR:
                self.delay_change_RM += 1
            else:
                print("distrib SR", list(self.prev_SR.values()))
                self.log(str(list(self.prev_SR.values())) + str(self.num_of_step))
                if all(np.array(list(self.prev_SR.values())) > 0.85):
                    self.SR_t += 1
                if self.SR_t >= self.SR_len: 
                    if not self.amount_angle_change >= self.max_amount_angle_change:
                        self.amount_angle_change += 1
                    elif not self.amount_radius_change >= self.max_amount_radius_change:
                        self.amount_radius_change += 1
                        self.amount_angle_change = 0
                    self.log("training mode up to " + str(self.training_mode) + " step: " + str(self.num_of_step) + " radius " + str(self.traning_radius) + "level " + str(self.level))
                    self.delay_change_RM = 0
                    self.SR_t = 0

    def _get_terminated(self, observation, RM):
        achievements = dict.fromkeys(self.reward_modes, False)
        if observation["current_dist_to_goal"] < 1.2:
            achievements["move"] = True
        if RM > 0 and achievements["move"] and abs(observation["orientation_error"]) < 15:
            achievements["rotation"] = True

        return achievements

    def get_reward(self, obs):
        achievements = self._get_terminated(obs, self.reward_mode)
        # print(achievements)
        terminated = False
        truncated = False
        punish_time = self._get_punish_time()

        if not achievements[self.reward_modes[0]]:
            reward = 2*punish_time
        else:
            if not achievements[self.reward_modes[1]]:
                reward = punish_time
            else:
                if self.reward_mode == 1:
                    terminated = True
                    reward = 5
                    return reward, terminated, truncated
                else:
                    print("error in get_reward function!")
                
        return reward, terminated, truncated
    
    def _get_punish_time(self):
        return -0.2#self._max_episode_length

    def move(self, action):
        # print("action type", type(action), action.shape, action)
        if not self.demonstrate:
            raw_forward = action[0]
            raw_angular = action[1]

            forward = (raw_forward + 1.0) / 2.0
            forward_velocity = forward * self.max_velocity

            angular_velocity = raw_angular * self.max_angular_velocity
        else:
            current_jetbot_position, current_jetbot_orientation = self.jetbot.get_world_pose()
            forward_velocity, angular_velocity = self.controle_module.pure_pursuit_controller(current_jetbot_position[0:2], self.current_jetbot_orientation,)
            # forward_velocity, angular_velocity = self.controle_module.get_PID_controle_velocity(current_jetbot_position[0:2], self.current_jetbot_orientation, self.goal_position)
            print("Agent get controle: ", forward_velocity, angular_velocity)
        for i in range(self._skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
            )
            self._my_world.step(render=False)
        
        return self.write_action_from_velocity(forward_velocity, angular_velocity)
    
    def write_action_from_velocity(self, forward_velocity, angular_velocity):
        action = None
        forward = forward_velocity / self.max_velocity
        raw_angular = angular_velocity / self.max_angular_velocity
        raw_forward = forward * 2.0 - 1.0
        action = np.array([[raw_forward, raw_angular]])
        #write action and bool in file
        # print("write action", action)
        return action
    
    def step(self, action):
        observations = self.get_observations()
        info = {}
        truncated = False
        terminated = False

        previous_jetbot_position, previous_jetbot_orientation = self.jetbot.get_world_pose()
        new_action = self.move(action)

        gt_observations = self.get_gt_observations(previous_jetbot_position, previous_jetbot_orientation)
        reward, terminated, truncated = self.get_reward(gt_observations)
        sources = ["time_out", "collision", "Nan"]
        source = "Nan"
        if not terminated:
            if self._is_timeout():
                truncated = True #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!здесь почему-то был false
                reward = -5
                source = sources[0]
            if self._is_collision() and self._get_current_time() > 2*self._skip_frame:
                truncated = True
                reward = -6
                source = sources[1]
        
        if terminated or truncated:
            if not self.demonstrate:
                self.get_success_rate(gt_observations, terminated, sources, source)
            self.start_step = True
            reward -= self._get_punish_time()
            if self.evalp and not self.demonstrate:
                s = 0
                if terminated:
                    s = 1
                self.eval_sr.append(s)
                if self.eval_write:
                    self.eval_write = 0
                    sr = sum(self.eval_sr) / len(self.eval_sr)
                    message = "r: " + str(round(self.traning_radius, 2)) + " a: " + str(round(self.traning_angle,2)) + " s: " + str(round(sr,2))
                    self.eval_sr = []
                    f = open(self.eval_log_path, "a+")
                    f.write(message + "\n")
                    f.close()
        print("gt_observations", gt_observations)
        message_to_collback = [self.demonstrate, new_action]
        # info["message_to_collback", message_to_collback]
        return observations, reward, terminated, truncated, info, message_to_collback
    
    def generate_random_key(self):
        """Генерирует случайный key в формате [k0, k1], где k0 и k1 — числа от 0 до 7."""
        k0 = 0
        k1 = 0
        r = r = asdict(self.config).get('eval_radius', None)
        test_level = asdict(self.config).get('test_level', None)
        level = self.level if not test_level else test_level
        print("current level is ", level)
        if self.level > 0:
            k0 = random.randint(0, 7)
            if self.level > 1:
                k1 = random.randint(0, 7)
        # return [3,2]
        return [k0, k1]
    
        # Новая функция для генерации JSON
    def generate_obstacles_json(self, obstacles_prop, goal_position, key):
        import json
        """Генерирует JSON-файл с информацией о препятствиях и цели, если файл еще не существует."""
        # Формируем имя файла на основе ключа
        key_str = f"{key[0]}_{key[1]}"
        output_file = os.path.join("/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/gjst", f"obstacles_and_goal_{key_str}.json")

        # Проверяем, существует ли файл
        if os.path.exists(output_file):
            print(f"JSON-файл для ключа {key} уже существует: {output_file}. Пропускаем генерацию.")
            return

        objects = []

        # Добавляем препятствия
        for idx, obstacle_prop in enumerate(obstacles_prop):
            # Позиция препятствия
            position = obstacle_prop["position"]
            # Центр ограничивающего куба (z = высота/2 = 0.7/2 = 0.35)
            bbox_center = [position[0], position[1], 0.35]
            # Размеры куба для препятствия
            bbox_extent = [0.6, 0.6, 0.7]
            # Описание (например, "table (0.60)")
            name = f"{obstacle_prop['shape']}"
            if name != "table":
                description = "obstacle"
            else:
                description = "soft obstacle"

            # Добавляем объект в список
            objects.append({
                "id": idx,
                "bbox_extent": bbox_extent,
                "bbox_center": bbox_center,
                "name": name,
                "description": description
            })

        # Добавляем цель
        # Центр цели (z = высота/2 = 0.7/2 = 0.35)
        bbox_center = [float(goal_position[0]), float(goal_position[1]), 0.35]
        # Размеры куба для цели
        bbox_extent = [0.2, 0.2, 0.7]
        # Описание цели
        name = "bowl (0.20)"
        description = "goal"

        # Добавляем цель в список с последним id
        objects.append({
            "id": len(obstacles_prop),
            "bbox_extent": bbox_extent,
            "bbox_center": bbox_center,
            "name": name,
            "description": description
        })

        # Сохраняем в JSON-файл
        try:
            with open(output_file, 'w') as f:
                json.dump(objects, f, indent=4)
            print(f"Saved obstacles and goal to {output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")

    def set_env(self, config, goal_position=np.array([1, 1, 0.0])):
        from omni.isaac.core.objects import VisualCuboid, FixedCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim
        from omni.isaac.core import World
        from .wheeled_robot import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.core.objects import VisualCuboid, FixedCuboid, VisualCylinder
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim
        import omni.isaac.core.utils.prims as prim_utils
        import omni.usd
        from pxr import UsdGeom, Gf, Semantics
        import random
        import json
        import os

        # Базовый путь к USD-файлам
        base_path = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/assets/scenes/obstacles"

        # Генерация конфигурации сцены
        self.key = self.generate_random_key()
        obstacles_prop = self.scene_controller.get_obstacles(key=self.key)
        scene_config = {
            "goal_position": goal_position.tolist(),
            "obstacles": obstacles_prop
        }

        # Сбрасываем позицию робота
        self.jetbot.set_world_pose([-10, -10, 0], get_quaternion_from_euler(0))
        self._my_world.reset()

        # Удаляем старые объекты
        stage = omni.usd.get_context().get_stage()
        # Удаляем старые препятствия
        i = 0
        while True:
            prim_path = f"/obstacle_{i}"
            if stage.GetPrimAtPath(prim_path).IsValid():
                delete_prim(prim_path)
                print(f"Deleted old obstacle: {prim_path}")
            else:
                break
            i += 1
        # Удаляем старую цель
        delete_prim("/cup")
        print("Deleted old cup")

        # Добавляем освещение только при инициализации
        if self.init:
            light_1 = prim_utils.create_prim(
                "/World/Light_1", "SphereLight", position=np.array([2.5, 5.0, 20.0]),
                attributes={"inputs:radius": 0.1, "inputs:intensity": 5e7, "inputs:color": (1.0, 1.0, 1.0)}
            )
            print("Added light at initialization")

        # Вызываем функцию для генерации JSON
        self.generate_obstacles_json(obstacles_prop, goal_position, self.key)

        # Создаём новые препятствия
        for i, obstacle_prop in enumerate(obstacles_prop):
            prim_path = f"/obstacle_{i}"
            if obstacle_prop["shape"] == "table":
                usd_path = f"{base_path}/table.usd"
                prim = stage.DefinePrim(prim_path, "Xform")
                prim.GetReferences().AddReference(usd_path)

                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(obstacle_prop["position"]))
                rotate_op = xform.AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3d(0, 0, 90))

                semantics_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                semantics_api.CreateSemanticTypeAttr().Set("class")
                semantics_api.CreateSemanticDataAttr().Set("table")
                print(f"Semantic tag for {prim_path}: {semantics_api.GetSemanticDataAttr().Get()}")

            elif obstacle_prop["shape"] == "chair":
                usd_path = random.choice([f"{base_path}/chair1.usd", f"{base_path}/chair2.usd"])
                prim = stage.DefinePrim(prim_path, "Xform")
                prim.GetReferences().AddReference(usd_path)

                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(obstacle_prop["position"]))
                rotate_op = xform.AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3d(0, 0, 0))

                semantics_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                semantics_api.CreateSemanticTypeAttr().Set("class")
                semantics_api.CreateSemanticDataAttr().Set("chair")
                print(f"Semantic tag for {prim_path}: {semantics_api.GetSemanticDataAttr().Get()}")

            elif obstacle_prop["shape"] == "trashcan":
                usd_path = f"{base_path}/TrashCan.usd"
                prim = stage.DefinePrim(prim_path, "Xform")
                prim.GetReferences().AddReference(usd_path)

                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(obstacle_prop["position"]))
                rotate_op = xform.AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3d(0, 0, 0))

                semantics_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                semantics_api.CreateSemanticTypeAttr().Set("class")
                semantics_api.CreateSemanticDataAttr().Set("trashcan")
                print(f"Semantic tag for {prim_path}: {semantics_api.GetSemanticDataAttr().Get()}")

            elif obstacle_prop["shape"] == "vase":
                usd_path = f"{base_path}/vase.usd"
                prim = stage.DefinePrim(prim_path, "Xform")
                prim.GetReferences().AddReference(usd_path)

                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(obstacle_prop["position"]))
                rotate_op = xform.AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3d(0, 0, 0))

                semantics_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                semantics_api.CreateSemanticTypeAttr().Set("class")
                semantics_api.CreateSemanticDataAttr().Set("vase")
                print(f"Semantic tag for {prim_path}: {semantics_api.GetSemanticDataAttr().Get()}")

        # Добавляем цель (чашу)
        create_prim(prim_path="/cup", position=goal_position, usd_path=asdict(config).get('cup_usd_path', None))
        cup_prim = stage.GetPrimAtPath("/cup")
        semantics_api = Semantics.SemanticsAPI.Apply(cup_prim, "Semantics")
        semantics_api.CreateSemanticTypeAttr().Set("class")
        semantics_api.CreateSemanticDataAttr().Set("bowl")
        print(f"Semantic tag for /cup: {semantics_api.GetSemanticDataAttr().Get()}")

        # Запускаем сбор данных для графа (если нужно)
        if False:
            scene_id = self.get_scene_id(scene_config)
            collector = GraphDataCollector(self._my_world, scene_config, self.output_dir, scene_name=f"scene_{scene_id}")
            collector.collect_data()
            
    def get_scene_id(self, scene_config):
        import json
        # Генерация уникального ID для конфигурации сцены
        import hashlib
        config_str = json.dumps(scene_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]  # Короткий хэш для читаемости
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.num_of_step > 0:
            self.change_reward_mode()
        self._my_world.reset()
        self.event = np.random.choice(self.events)
        self.num_of_step += 1

        self.goal_position, self.get_target_positions, self.num_of_envs = self.scene_controller.get_target_position(self.event, self.eval, self.evalp)

        correct_position = False
        tuning = asdict(self.config).get('tuning', None)
        r = asdict(self.config).get('eval_radius', None)
        if self.traning_radius > 2.5 + self.level*2:
            self.level += 1
        while not correct_position:
            eval = 1 if self.eval else 0
            add_r = 1 if tuning else 0
            if eval == add_r and eval == 1:
                print("u use eval and tuning")
            random_angle = 0
            self.traning_radius = add_r * r + r * eval + self.amount_radius_change * self.max_traning_radius / self.max_amount_radius_change
            # self.traning_radius = 2.8
            self.traning_angle = eval * random_angle + self.amount_angle_change * self.max_trining_angle / self.max_amount_angle_change
            print("radius is ", self.traning_radius, self.traning_angle)
            new_pos, new_angle, correct_position = self.scene_controller.get_robot_position(self.goal_position[0], self.goal_position[1], self.traning_radius, self.traning_angle)

        self.set_env(config=self.config, goal_position=self.goal_position)
        self.jetbot.set_world_pose(new_pos, get_quaternion_from_euler(new_angle))

        get_prob_true = lambda x: np.random.rand() <= x
        use_control = 1 if asdict(self.config).get('control', None) else 0
        self.imitation_part = 1 if asdict(self.config).get('imitation', None) else 0
        if (self.eval or self.evalp) and not self.imitation_part:
            self.demonstrate = get_prob_true(-1)
        else:
            self.demonstrate = use_control * get_prob_true(0.3 + 1 * self.imitation_part)
        print("demonstrate: ", self.demonstrate)
        if self.demonstrate:
            self.controle_module.update(new_pos, self.goal_position, self.get_target_positions, key=self.key)

        observations = self.get_observations()
        if self.use_graph:
            self.graph_embedding = self.graph_module.get_graph_embedding(self.num_of_envs)
        return observations, {}
    
    def get_path(self):
        pass

    def tensor_to_pil(self, image_tensor):
        # Убедимся, что тензор находится в диапазоне [0, 255]
        image_tensor = image_tensor.clamp(0, 255)
        
        # Преобразуем (C, H, W) → (H, W, C) и приводим к типу uint8
        image_np = image_tensor.permute(1, 2, 0).byte().cpu().numpy()
        
        # Создаем изображение
        return Image.fromarray(image_np, mode="RGB")
    
    def get_observations(self):
        self._my_world.render()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        images = self.pytorch_listener.get_rgb_data()
        # print(len(images))
        if images is not None:
            from torchvision.utils import save_image, make_grid
            img = images/255
            # img[0].resize((3, 128, 128))#, Image.ANTIALIAS)
            # print("len img", img[0].size())
            # if self._get_current_time() < 20:
            # save_image(make_grid(img[0], nrows = 2), '/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/Aloha_graph/Aloha/img/memory.png')
        else:
            print("Image tensor is NONE!")
        transform = T.ToPILImage()
        img_current_0 = self.clip_preprocess(transform(img[0])).unsqueeze(0).to(self.device)
        img_current_1 = self.clip_preprocess(transform(img[1])).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_current_emb_0 = self.clip_model.encode_image(img_current_0)
            img_current_emb_1 = self.clip_model.encode_image(img_current_1)
        event = self.event,
        if event == 1:
            s = "go to the bowl wall with 1 color"
        else:
            s = "go to the bowl wall with 2 color"

        text = clip.tokenize([s]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        # graph_embedding = self.get_graph_embedding()!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # print("embedding ", type(graph_embedding))
        # print(img[0])
        self.get_graph_embedding(key=self.key)
        if self.use_graph:
            return np.concatenate(
            [
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                # self.img_goal_emb[0].cpu(),
                img_current_emb_0[0].cpu(),
                img_current_emb_1[0].cpu(),
                text_features[0].cpu(),
                # mem[0].cpu(),
                self.graph_embedding.cpu().detach().numpy(),
            ]
        )
        if self.use_memory:
            self.memory.add(self.tensor_to_pil(img[0]))
            mem = self.memory.get_embedding()
            self.memory.save_memory_as_grid()
            return np.concatenate(
            [
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                # self.img_goal_emb[0].cpu(),
                img_current_emb_0[0].cpu(),
                img_current_emb_1[0].cpu(),
                text_features[0].cpu(),
                mem[0].cpu(),
                # graph_embedding.cpu().detach().numpy(),
            ]
        )
        return np.concatenate(
            [
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                # self.img_goal_emb[0].cpu(),
                img_current_emb_0[0].cpu(),
                img_current_emb_1[0].cpu(),
                text_features[0].cpu(),
                # graph_embedding.cpu().detach().numpy(),
            ]
        )

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
    
    def log(self, message):
        f = open(self.log_path, "a+")
        f.write(message + "\n")
        f.close()
        return

    def _on_contact_report_event(self, contact_headers, contact_data):
        from pxr import PhysicsSchemaTools

        for contact_header in contact_headers:
            # instigator
            act0_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            # recipient
            act1_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
            # the specific collision mesh that belongs to the Rigid Body
            cur_collider = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))

            # iterate over all contacts
            contact_data_offset = contact_header.contact_data_offset
            num_contact_data = contact_header.num_contact_data
            for index in range(contact_data_offset, contact_data_offset + num_contact_data, 1):
                cur_contact = contact_data[index]

                # find the magnitude of the impulse
                cur_impulse =  cur_contact.impulse[0] * cur_contact.impulse[0]
                cur_impulse += cur_contact.impulse[1] * cur_contact.impulse[1]
                cur_impulse += cur_contact.impulse[2] * cur_contact.impulse[2]
                cur_impulse = math.sqrt(cur_impulse)
            if num_contact_data > 1 or not self.scene_controller.no_intersect_with_obstacles(self.current_jetbot_position): #1 contact with flore here yet
                self.collision = True

    def get_env_info(self):  # <-- Этот метод нужен rl_games!
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": 1  # Количество агентов
        }
    
    def get_graph_embedding(self, key):
        """
        Возвращает эмбеддинг сцены для заданного key, используя кэш последнего эмбеддинга.
        
        Args:
            key (list): Ключ в формате [k0, k1], например [3, 1].
        
        Returns:
            torch.Tensor: Тензор формы (max_objects, 390) с эмбеддингом сцены.
        """
        # Проверяем, есть ли кэш и совпадает ли текущий key с последним использованным
        if hasattr(self, 'cache') and self.cache is not None:
            cached_key, cached_embedding = self.cache
            if cached_key == key:
                # print(f"Using cached embedding for key {key}")
                return cached_embedding

        # Формируем путь к JSON-файлу на основе key
        key_str = f"{key[0]}_{key[1]}"  # Например, "3_1"
        json_file = os.path.join("/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/gjst/", f"obstacles_and_goal_{key_str}.json")

        # Проверяем, существует ли JSON-файл
        if not os.path.exists(json_file):
            print(f"JSON file for key {key} not found at {json_file}!")
            return torch.zeros((8, 390), dtype=torch.float32, device=self.device)

        # Генерируем эмбеддинг с помощью get_simple_embedding_from_json
        try:
            scene_embeddings = get_simple_embedding_from_json(
                json_file, max_objects=8, device=self.device
            )
            # Сохраняем в кэш
            self.cache = (key, scene_embeddings)
            print(f"Generated and cached embedding for key {key}")
            return scene_embeddings
        except Exception as e:
            print(f"Error generating embedding for key {key}: {e}")
            return torch.zeros((8, 390), dtype=torch.float32, device=self.device)
    



    #part with graph, should to move it

import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def get_simple_embedding_from_json(json_file, model_name='all-MiniLM-L6-v2', max_objects=8, device=None):
    """
    Генерирует эмбеддинги для всех объектов из JSON-файла с фиксированной размерностью.
    
    Args:
        json_file (str): Путь к JSON-файлу с описанием сцены.
        model_name (str): Название модели sentence-transformers для генерации текстовых эмбеддингов.
        max_objects (int): Максимальное количество объектов, которое будет включено в итоговый тензор.
        device (str or torch.device): Устройство для вычислений (например, 'cuda:0' или 'cpu').
    
    Returns:
        torch.Tensor: Тензор формы (max_objects, 390), где каждый объект представлен отдельно.
                     Если объектов меньше max_objects, оставшиеся места заполняются нулями.
                     Если объектов больше max_objects, лишние обрезаются.
    """
    # Определяем устройство
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Загружаем модель sentence-transformers
    model = SentenceTransformer(model_name, device=device)

    # Читаем JSON-файл
    with open(json_file, "r") as f:
        data = json.load(f)

    object_features = []

    # Обрабатываем каждый объект в JSON
    for obj_data in data:
        # Извлекаем числовые характеристики и перемещаем на нужное устройство
        bbox_extent = torch.tensor(obj_data["bbox_extent"], dtype=torch.float32, device=device)  # 3 dimensions
        bbox_center = torch.tensor(obj_data["bbox_center"], dtype=torch.float32, device=device)  # 3 dimensions

        # Извлекаем текстовое описание и генерируем эмбеддинг
        description = obj_data["description"]
        text_embedding = model.encode(description, convert_to_tensor=True, device=device)  # 384 dimensions (для all-MiniLM-L6-v2)

        # Объединяем числовые характеристики и текстовый эмбеддинг
        features = torch.cat([bbox_extent, bbox_center, text_embedding], dim=0)  # 3 + 3 + 384 = 390 dimensions
        object_features.append(features)

    # Если объектов нет, возвращаем тензор с нулями
    if not object_features:
        return torch.zeros((max_objects, 390), dtype=torch.float32, device=device)

    # Преобразуем список в тензор
    object_tensor = torch.stack(object_features)  # Shape: (num_objects, 390)

    # Фиксируем размерность
    num_objects = object_tensor.shape[0]
    feature_dim = object_tensor.shape[1]  # 390

    # Создаем итоговый тензор фиксированной формы (max_objects, 390)
    fixed_tensor = torch.zeros((max_objects, feature_dim), dtype=torch.float32, device=device)

    if num_objects <= max_objects:
        # Если объектов меньше или равно max_objects, заполняем тензор и оставляем нули в конце
        fixed_tensor[:num_objects, :] = object_tensor
    else:
        # Если объектов больше max_objects, обрезаем лишние
        fixed_tensor = object_tensor[:max_objects, :]

    return fixed_tensor

# # Пример использования
# json_file = "/path/to/your/scene_graph_2_26/1.json"  # Укажите путь к вашему JSON-файлу
# scene_embeddings = get_simple_embedding_from_json(json_file, max_objects=10)
# print("Scene embeddings shape:", scene_embeddings.shape)
# print("Scene embeddings:", scene_embeddings)