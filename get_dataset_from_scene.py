from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import matplotlib.pyplot as plt
import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
import os
import omni.replicator.core as rep
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import imageio
import cv2
from scipy.spatial.transform import Rotation as R
import json
from omni.kit.viewport.utility import get_active_viewport

# Параметры симуляции
physics_dt = 1.0 / 20.0
rendering_dt = 1.0 / 20.0
my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()


asset_path = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/assets/scenes/scenes_sber_kitchen_for_BBQ/room_for_test_loc.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/env")

# Параметры для случайных позиций
N_POSITIONS = 100  # Количество случайных позиций камеры
CAMERA_HEIGHT = 1.1  # Высота камеры (уровень глаз человека)
CAMERA_RADIUS = 0.3  # Минимальное расстояние до препятствия (в метрах)
BOUNDARY_EXTENSION = 1.0  # Расширение границ на 1 метр
JSON_PATH = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/test_graph.json"   # Укажите путь к JSON-файлу с препятствиями

# Чтение JSON с препятствиями
with open(JSON_PATH, 'r') as f:
    obstacles = json.load(f)

# Извлечение координат bbox_center
obstacle_positions = np.array([obj["bbox_center"][:2] for obj in obstacles])  # Берем только x, y

# Определение границ области
min_coords = np.min(obstacle_positions, axis=0)
max_coords = np.max(obstacle_positions, axis=0)
min_x, min_y = min_coords - BOUNDARY_EXTENSION
max_x, max_y = max_coords + BOUNDARY_EXTENSION
print(f"Computed boundaries: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")

# Функция для создания матрицы трансформации
def transformation_matrix(position, orientation):
    w, x, y, z = orientation
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = position
    return transformation

# Настройка камеры
# Настройка камеры
camera = Camera(prim_path="/World/Camera")
my_world.reset()
camera.initialize()
camera.add_distance_to_camera_to_frame()

# Устанавливаем разрешение камеры через render_product
camera_resolution = (1280, 720)  # Задайте нужное разрешение, например, (1920, 1080)
render_product = rep.create.render_product(camera.prim_path, resolution=camera_resolution)

# Устанавливаем начальную позицию камеры
camera.set_local_pose(np.array([0, 0, CAMERA_HEIGHT]), rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True), camera_axes="world")
camera_prim = get_current_stage().GetPrimAtPath("/World/Camera")
camera_prim.GetAttribute("horizontalAperture").Set(80)

# Генерируем случайные позиции камеры, избегая препятствий
np.random.seed(42)  # Для воспроизводимости
positions = []
while len(positions) < N_POSITIONS:
    candidate_pos = np.random.uniform(
        low=[min_x, min_y, CAMERA_HEIGHT],
        high=[max_x, max_y, CAMERA_HEIGHT]
    )
    distances = np.linalg.norm(obstacle_positions - candidate_pos[:2], axis=1)
    min_distance = np.min(distances)
    if min_distance > CAMERA_RADIUS:
        positions.append(candidate_pos)
        print(f"Added position {len(positions)}: {candidate_pos}, min distance to obstacle: {min_distance:.2f} m")
    else:
        print(f"Rejected position {candidate_pos}, min distance to obstacle: {min_distance:.2f} m")

positions = np.array(positions)

# Генерируем случайные углы поворота (yaw) для горизонтального взгляда
yaw_angles = np.random.uniform(low=0, high=2 * np.pi, size=N_POSITIONS)

# Вычисляем ориентации
orientations = []
for pos, yaw in zip(positions, yaw_angles):
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # Горизонтальное направление
    target = pos + direction
    # Вычисляем ориентацию вручную, используя направление (direction) и преобразуем в кватернион
    # Предполагаем, что камера смотрит горизонтально (roll=0, pitch=0, только yaw)
    yaw = np.arctan2(direction[1], direction[0])  # Вычисляем угол yaw
    euler_angles = np.array([0, 0, yaw])  # roll, pitch, yaw (в радианах)
    orientation = rot_utils.euler_angles_to_quats(euler_angles, degrees=False)
    orientations.append(orientation)

# Настройка путей для сохранения данных
scene_name = "test_scene"

OUTPUT_DIR_RGB = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/rgb_dataset"  # Папка для RGB-изображений
OUTPUT_DIR_DEPTH = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/d_dataset"  # Папка для данных глубины
base_dir = f"/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/dataset_info"
traj_dir = f"/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/dataset_info"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(OUTPUT_DIR_RGB, exist_ok=True)
os.makedirs(OUTPUT_DIR_DEPTH, exist_ok=True)
camera_poses = {}
traj_file_path = os.path.join(traj_dir, "traj.txt")
if not os.path.exists(traj_file_path):
    with open(traj_file_path, 'w') as file:
        pass
    print(f"File '{traj_file_path}' has been created.")

# Инициализация симуляции
for j in range(10):
    camera.set_local_pose(positions[0], orientations[0], camera_axes="world")
    my_world.step(render=True)
    simulation_app.update()

# Основной цикл сбора данных
# Основной цикл сбора данных
j = 0
while simulation_app.is_running() and j < len(positions):
    pos = positions[j]
    ori = orientations[j]
    print(f"Processing position {j+1}/{len(positions)}: {pos}, yaw={np.degrees(yaw_angles[j]):.1f} degrees")
    
    # Устанавливаем позицию и ориентацию камеры
    camera.set_local_pose(pos, ori, camera_axes="world")
    
    # Даём симуляции несколько шагов для стабилизации
    for k in range(10):
        my_world.step(render=True)

    # Получаем данные с камеры (используем render_product камеры)
    depth_annotators = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annotators.attach([render_product])
    depth_image = depth_annotators.get_data()

    rgb_annotators = rep.AnnotatorRegistry.get_annotator("LdrColor")
    rgb_annotators.attach([render_product])
    rgba_image = rgb_annotators.get_data()

    seg_annotators = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    seg_annotators.attach([render_product])
    seg_data = seg_annotators.get_data()
    seg_info = seg_data['info']['idToLabels']
    seg_image = seg_data['data'].astype(np.uint8)

    # Сохранение данных
    img_path = os.path.join(OUTPUT_DIR_RGB, f"frame{j:06d}.jpg")
    depth_path = os.path.join(OUTPUT_DIR_DEPTH, f"depth{j:06d}.png")
    seg_path = os.path.join(base_dir, f"seg{j:06d}.png")
    seg_info_path = os.path.join(base_dir, f"seg{j:06d}_info.json")


    position_, orientation_ = camera.get_world_pose(camera_axes="world")
    camera_poses[f"frame{j:06d}.jpg"] = {
        "position": position_.tolist(),
        "orientation": orientation_.tolist()
    }

    min_val, max_val = 0.01, 10.0
    if depth_image.size != 0 and rgba_image.size != 0:
        # Сохранение глубины
        clipped_depth = np.clip(depth_image, min_val, max_val)
        normalized_depth = ((clipped_depth - min_val) / (max_val - min_val)) * 65535
        depth_image_uint16 = normalized_depth.astype("uint16")
        cv2.imwrite(depth_path, depth_image_uint16)
        print(f"Saved Depth data to {depth_path}")

        # Сохранение сегментации
        cv2.imwrite(seg_path, seg_image)
        print(f"Saved Segmentation data to {seg_path}")

        # Сохранение RGB
        rgb = rgba_image[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, bgr)
        print(f"Saved RGB image to {img_path}")

        # Сохранение матрицы трансформации
        position_, orientation_ = camera.get_world_pose(camera_axes="world")
        transformation_matrix_result = transformation_matrix(position_, orientation_)
        # with open(traj_file_path, "a") as traj_file:
        #     transform_str = ' '.join(map(str, transformation_matrix_result.flatten()))
        #     traj_file.write(transform_str + "\n")
        
        # # Сохранение информации о сегментации
        # with open(seg_info_path, "w") as json_file:
        #     json.dump(seg_info, json_file, indent=4)
        #     print(f"Saved Segmentation info to {seg_info_path}")
    
    j += 1
with open(os.path.join(base_dir, "camera_poses.json"), "w") as json_file:
    json.dump(camera_poses, json_file, indent=4)
# Закрытие симуляции
simulation_app.close()