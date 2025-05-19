# graph_data_collector.py
import numpy as np
import os

import cv2
import json
class GraphDataCollector:
    def __init__(self, world, scene_config, output_dir, scene_name="scene"):
        self.world = world  # Используем существующий World из CLGRCENV
        self.scene_config = scene_config  # Конфигурация сцены
        self.output_dir = output_dir
        self.scene_name = scene_name
        self.base_dir = os.path.join(output_dir, scene_name, "results")
        self.traj_dir = os.path.join(output_dir, scene_name)
        os.makedirs(self.base_dir, exist_ok=True)
        self.traj_file_path = os.path.join(self.traj_dir, "traj.txt")
        print("json file will bw saved in ", self.traj_dir), 
        if not os.path.exists(self.traj_file_path):
            with open(self.traj_file_path, 'w') as file:
                pass

    def transformation_matrix(self, position, orientation):
        w, x, y, z = orientation
        rotation_matrix = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
        ])
        # correction = np.array([
        #     [1,  0,   0  ],
        #     [0,  0,   1  ],
        #     [0, -1,   0  ]
        # ])
        # rotation_matrix = correction @ rotation_matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = position
        return transformation

    def interpolate_keyframes_with_euler(self, keyframes, i):
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        for j in range(len(keyframes) - 1):
            t0, t1 = keyframes[j]['time'], keyframes[j + 1]['time']
            if t0 <= i <= t1:
                kf0, kf1 = keyframes[j], keyframes[j + 1]
                break
        else:
            return None, None
        alpha = (i - t0) / (t1 - t0)
        next_translation = (1 - alpha) * np.array(kf0['translation']) + alpha * np.array(kf1['translation'])
        euler0 = kf0['euler_angles']
        euler1 = kf1['euler_angles']
        interpolated_euler = (1 - alpha) * np.array(euler0) + alpha * np.array(euler1)
        next_orientation = euler_angles_to_quat(interpolated_euler, degrees=True)
        return next_translation, next_orientation

    def collect_data(self):
        # Настройка камеры
        from omni.isaac.core import World
        from omni.isaac.sensor import Camera
        from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
        import omni.isaac.core.utils.numpy.rotations as rot_utils
        import omni.replicator.core as rep

        camera = Camera(prim_path="/World/Camera")
        camera.initialize()
        camera.add_distance_to_camera_to_frame()
        camera.set_local_pose(np.array([3, 0, 1.5]), rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True), camera_axes="world")
        camera_prim = get_current_stage().GetPrimAtPath("/World/Camera")
        camera_prim.GetAttribute("horizontalAperture").Set(80)

        # Задаём разрешение камеры через рендер-продукт
        camera_resolution = (1280, 720)  # Укажи нужное разрешение здесь, например, (1280, 720) или (1920, 1080)
        render_product = rep.create.render_product(camera.prim_path, resolution=camera_resolution)

        # Ключевые кадры для траектории камеры
        keyframes_move = [
            {'time': 0, 'translation': [0, 3, 2.2], 'euler_angles': [0, 30, 0]},
            {'time': 200, 'translation': [5.5, 3, 2.2], 'euler_angles': [0, 30, 0]},
            {'time': 300, 'translation': [5.5, 3, 2.2], 'euler_angles': [0, 30, -180]},
            {'time': 500, 'translation': [1.5, 3, 2.2], 'euler_angles': [0, 30, -180]},
        ]
        record_keyframe = keyframes_move

        # Инициализация
        for j in range(50):
            next_translation, next_orientation = self.interpolate_keyframes_with_euler(record_keyframe, 0)
            if j == 0:
                camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
            self.world.step(render=True)

        # Основной цикл сбора данных
        i = 0
        j = 0
        step_size = 2
        while True:
            next_translation, next_orientation = self.interpolate_keyframes_with_euler(record_keyframe, i)
            if next_translation is None:
                break
            camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
            i += step_size

            for k in range(10):
                self.world.step(render=True)

            # Используем созданный render_product вместо camera._render_product_path
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
            img_path = os.path.join(self.base_dir, f"frame{j:06d}.jpg")
            depth_path = os.path.join(self.base_dir, f"depth{j:06d}.png")
            seg_path = os.path.join(self.base_dir, f"seg{j:06d}.png")
            seg_info_path = os.path.join(self.base_dir, f"seg{j:06d}_info.json")

            min_val, max_val = 0.01, 10.0
            if depth_image.size != 0 and rgba_image.size != 0:
                clipped_depth = np.clip(depth_image, min_val, max_val)
                normalized_depth = ((clipped_depth - min_val) / (max_val - min_val)) * 65535
                depth_image_uint16 = normalized_depth.astype("uint16")
                cv2.imwrite(depth_path, depth_image_uint16)
                cv2.imwrite(seg_path, seg_image)
                rgb = rgba_image[:, :, :3]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, bgr)

                position_, orientation_ = camera.get_world_pose(camera_axes="world")
                transformation_matrix_result = self.transformation_matrix(position_, orientation_)
                with open(self.traj_file_path, "a") as traj_file:
                    transform_str = ' '.join(map(str, transformation_matrix_result.flatten()))
                    traj_file.write(transform_str + "\n")
                with open(seg_info_path, "w") as json_file:
                    json.dump(seg_info, json_file, indent=4)
            j += 1

        # Очистка камеры после сбора данных
        from omni.isaac.core.utils.prims import delete_prim
        delete_prim("/World/Camera")