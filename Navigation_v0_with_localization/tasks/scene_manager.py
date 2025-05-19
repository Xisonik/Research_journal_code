import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

from configs.main_config import MainConfig

d = Path().resolve()#.parent
general_path = str(d) + "/standalone_examples/Aloha_graph/Aloha"
log = general_path + "/logs/"

import random

class Scene_controller:
    def __init__(self):
        self.config = MainConfig()
        self.targets = []
        self.obstacles = []
        self.change_line = 0
        self.repeat = 5
        self.obstacles_shape = ["table", "chair", "human"]
        self.generate_positions_for_contole_module()
        self._init_scene()
        self.robot_r = 0.34

    def generate_positions_for_contole_module(self, key=[-1,-1]):
        self.obstacles = []
        # Добавляем стол (без изменений)
        self.obstacles.append(self._set_obstacle(
            shape="table",
            position=[0.5, 2.5, 0.0],
            len_x=0.5,
            len_y=2.5
        ))

        # Создаём двумерный массив 3x2, заполненный нулями
        grid = np.zeros((2, 3), dtype=int)

        # Заполняем строки массива grid на основе key
        for row in range(2):  # Ограничиваем 2 строками
            if key[row] == -1:
                # Случайное заполнение строки
                n_ones = np.random.randint(0, 4)  # От 0 до 3 единиц
                positions = random.sample(range(3), n_ones)
                grid[row, positions] = 1
            else:
                # Преобразуем число из key в двоичный вид (3 бита)
                if 0 <= key[row] <= 7:
                    grid[row] = [int(bit) for bit in format(key[row], '03b')]
                else:
                    raise ValueError(f"Число в key[{row}] должно быть от 0 до 7")

        # Преобразуем единицы в препятствия (стулья)
        y_values = [1.5, 2.5, 3.5]  # Соответствие столбцов y-координатам
        for row in range(2):
            x = 2.5 + row*2  # Строка 0 → x=2, строка 1 → x=4
            for col in range(3):
                if grid[row, col] == 1:
                    self.obstacles.append(self._set_obstacle(
                        shape="chair",
                        position=[x, y_values[col], 0.0]
                    ))
        return # Задаём позиции и типы препятствий
        
    def _init_scene(self):
        pass
    
    def get_obstacles(self,key=[-1,-1]):
        self.generate_positions_for_contole_module(key)

        return self.obstacles    

    def _set_obstacle(self, shape, position, usd_path=None, radius=0, len_x=0.5, len_y=0.5, height=0.5):
        return {
            "shape": shape,
            "position": position,
            "radius": radius,
            "len_x": len_x,
            "len_y": len_y,
            "height": height
        }

    def no_intersect_with_obstacles(self, robot_pos, add_r=0):
        no_intersect = True
        
        for obstacle in self.obstacles:
            if obstacle["shape"] == "table":
                if (np.abs(obstacle["position"][0] - robot_pos[0]) < (self.robot_r + obstacle["len_x"] + add_r) and 
                    np.abs(obstacle["position"][1] - robot_pos[1]) < (self.robot_r + obstacle["len_y"] + add_r)):
                    no_intersect = False
            elif obstacle["shape"] in ["chair", "human"]:
                # Предполагаем радиус столкновения 0.5 м для стула и человека
                if np.abs(np.linalg.norm(np.array(obstacle["position"][0:2]) - robot_pos)) < (self.robot_r + 0.4 + add_r):
                    no_intersect = False
        
        return no_intersect

    def get_target_position(self, event, eval, evalp):
        poses_bowl = [np.array([7.5, 1, 0.7]),
                      np.array([7.5, 0.8, 0.7]),
                      np.array([7.5, 1.2, 0.7]),
                      np.array([0.5, 4.8, 0.7]),
                      np.array([0.5, 2.5, 0.7]),
                      np.array([0.5, 5, 0.7])]
        
        if event == 0:
            num_of_envs = np.random.choice([1])
        elif event == 1:
            num_of_envs = np.random.choice([4])

        if not eval and not evalp:
            goal_position = poses_bowl[num_of_envs]
        else:
            goal_position = poses_bowl[num_of_envs]

        goal_position = poses_bowl[num_of_envs]
        goal_position = np.array([0.5, 2.5, 0.7])
        poses_bowl = [goal_position]
        num_of_envs = 0
        return goal_position, poses_bowl, num_of_envs

    def get_robot_position(self, x_goal, y_goal, traning_radius=0, traning_angle=0, tuning=0):
        # return [4,3,0], -np.pi
        traning_radius_start = 1.2
        k = 0
        self.change_line += 1
        reduce_r = 1
        reduce_phi = 1
        track_width = 1.2
        
        # if self.change_line >= self.repeat:
        #     reduce_r = np.random.rand()
        #     reduce_phi = np.random.rand()
        #     self.change_line=0
        # print("reduce", reduce_r)
        while True:
            k += 1
            robot_pos = np.array([x_goal, y_goal, 0.1])

            alpha = np.random.rand()*2*np.pi
            robot_pos += (traning_radius_start+reduce_r*(traning_radius)) * np.array([np.cos(alpha), np.sin(alpha), 0])

            goal_world_position = np.array([x_goal, y_goal])
            nx = np.array([-1,0])
            ny = np.array([0,1])
            to_goal_vec = goal_world_position - robot_pos[0:2]
            
            cos_angle = np.dot(to_goal_vec, nx) / np.linalg.norm(to_goal_vec) / np.linalg.norm(nx)
            n = np.random.randint(2)
            
            quadrant = self._get_quadrant(nx, ny, to_goal_vec)
            # return [5.8,2.5,0], np.pi/2, True#quadrant*np.arccos(cos_angle) + ((-1)**n)*reduce_phi*traning_angle, True
            if self.no_intersect_with_obstacles(robot_pos[0:2], 0.1) and robot_pos[0] < 5 and robot_pos[0] > 1 and robot_pos[1] < 5 and robot_pos[1] > 1:
                n = np.random.randint(2)
                return robot_pos, quadrant*np.arccos(cos_angle) + ((-1)**n)*reduce_phi*traning_angle, True
            elif k >= 1000:
                print("can't get correct robot position: ", robot_pos, quadrant*np.arccos(cos_angle) + reduce_phi*traning_angle, reduce_r*traning_radius)
                return 0, 0, False
                  
    def _get_quadrant(self, nx, ny, vector):
        LR = vector[0]*nx[1] - vector[1]*nx[0]
        mult = 1
        if LR < 0:
            mult = -1
        return mult

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