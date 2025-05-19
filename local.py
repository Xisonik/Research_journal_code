import json
import numpy as np
from scipy.spatial import ConvexHull
import cv2
import os
import logging
import matplotlib.pyplot as plt
from ultralytics import YOLO
import uuid
from itertools import combinations
from scipy.optimize import minimize
import re

class RobotLocalization:
    def __init__(self, rgb_dir, depth_dir, graph_path, output_dir):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.graph_path = graph_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Configure logging
        log_file = os.path.join(output_dir, 'localization.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        # logging.info(f"Initialized RobotLocalization for RGB dir: {rgb_dir}, Depth dir: {depth_dir}")

        self.model = YOLO('yolov8m-seg.pt')
        self.scene = []
        self.obj_detect = []
        self.seq_iobj = []
        self.closest_object = None
        self.closest_detected_set = None
        self.robot_position = None
        self.robot_orientation = None
        self.objects = []
        self.unique_id_map = {}
        self.grid_size = 18
        self.inner_grid_size = 16
        self.distance_ratios = []
        self.depths = []
        self.local_graph = []
        self.rgb_images = []
        self.depth_images = []
        self.frame_num = 0

    def get_image_pairs(self):
        """Retrieve paired RGB and depth images."""
        rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.startswith('frame') and f.endswith('.jpg')])
        self.rgb_images = [os.path.join(self.rgb_dir, f) for f in rgb_files]
        self.depth_images = []
        for rgb_file in rgb_files:
            frame_num = re.search(r'frame(\d+)\.jpg', rgb_file).group(1)
            depth_file = f'depth{frame_num}.png'
            depth_path = os.path.join(self.depth_dir, depth_file)
            if os.path.exists(depth_path):
                self.depth_images.append(depth_path)
            else:
                logging.warning(f"Depth image {depth_file} not found for {rgb_file}")
                self.rgb_images.remove(os.path.join(self.rgb_dir, rgb_file))
        # logging.info(f"Found {len(self.rgb_images)} paired RGB and depth images")
        return len(self.rgb_images) > 0

    def load_graph(self):
        """Load and parse the knowledge graph JSON."""
        with open(self.graph_path, 'r') as f:
            graph = json.load(f)
        # logging.info(f"Loaded graph with {len(graph)} objects")
        
        id_counts = {}
        for obj in graph:
            obj_id = str(obj['id'])
            if obj_id in id_counts:
                id_counts[obj_id] += 1
                new_id = str(uuid.uuid4())
                self.unique_id_map[obj_id] = new_id
                obj['id'] = new_id
            else:
                id_counts[obj_id] = 1
                self.unique_id_map[obj_id] = obj_id
            self.objects.append(obj)
        return graph

    def build_scene_grid(self):
        """Build 2D scene grid from graph objects."""
        graph = self.load_graph()
        centers = [obj['bbox_center'][:2] for obj in graph]
        ids = [str(obj['id']) for obj in graph]
        
        centers_array = np.array(centers)
        min_coords = np.min(centers_array, axis=0)
        max_coords = np.max(centers_array, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1
        
        self.scene = [[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for i, (center, obj_id) in enumerate(zip(centers, ids)):
            x, y = center
            scaled_x = (x - min_coords[0]) / range_coords[0] * (self.inner_grid_size - 1) + 1
            scaled_y = (y - min_coords[1]) / range_coords[1] * (self.inner_grid_size - 1) + 1
            grid_x = int(round(scaled_x))
            grid_y = int(round(self.grid_size - 1 - scaled_y))
            grid_x = max(0, min(self.grid_size - 1, grid_x))
            grid_y = max(0, min(self.grid_size - 1, grid_y))
            self.scene[grid_y][grid_x].append(obj_id)
               
        names = sorted(set(obj['name'] for obj in graph))
        self.obj_detect = []
        for name in names:
            for obj in graph:
                if obj['name'] == name:
                    is_goal = 1 if obj['description'] == 'goal' else 0
                    self.obj_detect.append([name, str(obj['id']), 0, is_goal])
        logging.info(f"OBJ_DETECT: {self.obj_detect}")

    def build_local_grid(self):
        """Build a small 8x8 grid for the local graph with the robot at (0, 0)."""
        grid_size = 8
        local_grid = [['' for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Нормализация координат объектов
        points = [obj['bbox_center'][:2] for obj in self.local_graph]
        points.append([0, 0])  # Добавляем позицию робота
        points = np.array(points)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1
        
        # Размещение объектов
        for obj in self.local_graph:
            x, y = obj['bbox_center'][:2]
            scaled_x = (x - min_coords[0]) / range_coords[0] * (grid_size - 1)
            scaled_y = (y - min_coords[1]) / range_coords[1] * (grid_size - 1)
            grid_x = int(round(scaled_x))
            grid_y = int(round(grid_size - 1 - scaled_y))
            grid_x = max(0, min(grid_size - 1, grid_x))
            grid_y = max(0, min(grid_size - 1, grid_y))
            name_short = obj['name'][:3]
            local_grid[grid_y][grid_x] = f"{name_short}_{obj['id'][-4:]}"
        
        # Размещение робота в (0, 0)
        robot_x, robot_y = 0, 0
        scaled_x = (robot_x - min_coords[0]) / range_coords[0] * (grid_size - 1)
        scaled_y = (robot_y - min_coords[1]) / range_coords[1] * (grid_size - 1)
        grid_x = int(round(scaled_x))
        grid_y = int(round(grid_size - 1 - scaled_y))
        grid_x = max(0, min(grid_size - 1, grid_x))
        grid_y = max(0, min(grid_size - 1, grid_y))
        local_grid[grid_y][grid_x] = 'R'
               
        return local_grid
    
    def plot_local_graph(self, local_grid, filename):
        """Plot the local graph grid."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"Local Graph ({filename})")
        rows, cols = len(local_grid), len(local_grid[0])
        
        # Рисуем линии сетки
        for i in range(rows + 1):
            ax.axhline(i, color='black', linewidth=1)
        for j in range(cols + 1):
            ax.axvline(j, color='black', linewidth=1)
        
        # Добавляем объекты и робота
        for i in range(rows):
            for j in range(cols):
                if local_grid[i][j]:
                    ax.text(
                        j + 0.5, rows - i - 0.5, local_grid[i][j],
                        ha='center', va='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
                    )
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(range(cols + 1))
        ax.set_yticks(range(rows + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, f'local_grid_{filename}.png'))
        plt.close()
        logging.info(f"Saved local graph plot: local_grid_{filename}.png")

    def detect_objects(self, rgb_path, depth_path):
        """Detect objects and compute depths using RGB and depth images."""
        results = self.model(rgb_path)
        detections = results[0].boxes
        centers = []
        names = []
        masks = []
        self.depths = []
        
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype(np.float32)
        min_val = 0.01
        max_val = 10
        depth_img = min_val + (depth_img / 65535.0) * (max_val - min_val)
        if depth_img is None:
            logging.error(f"Failed to load depth image: {depth_path}")
            return False

        masks_dir = os.path.join(self.output_dir, f'masks_{self.frame_num}')
        os.makedirs(masks_dir, exist_ok=True)
        # print("depths ", len(names), len(self.depths))
        for i, box in enumerate(detections):
            name = self.model.names[int(box.cls)]
            # print("name ", name)
            xyxy = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) / 2
            # print("xcenter_x ", center_x)
            centers.append(center_x)
            names.append(name)
            
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                mask = results[0].masks[i].data.cpu().numpy().squeeze()
                mask = cv2.resize(mask, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)
                depth_values = depth_img[mask]
                if depth_values.size > 0:
                    depth_min = depth_values.min()
                    depth_max = depth_values.max()
                    depth_mean = depth_values.mean()
                    self.depths.append(depth_min)
                    # print("depth_min ", depth_min)
                else:
                    self.depths.append(0)
                    logging.warning(f"No depth values for object {name} (index {i})")
                mask_img = (mask * 255).astype(np.uint8)
                mask_filename = os.path.join(masks_dir, f"mask_{i}_{name}.png")
                cv2.imwrite(mask_filename, mask_img)
            else:
                self.depths.append(0)
                logging.warning(f"No mask available for object {name} (index {i})")

        # print("depths ", len(name), len(names), len(self.depths))
        scene_types = {obj['name'] for obj in self.objects}
        filtered_indices = [i for i, name in enumerate(names) if name in scene_types]
        if not filtered_indices:
            logging.error("No detected objects match the scene graph")
            return False
        filtered_names = [names[i] for i in filtered_indices]
        filtered_centers = [centers[i] for i in filtered_indices]
        self.depths = [self.depths[i] for i in filtered_indices]
        
        sorted_indices = np.argsort(filtered_centers)
        self.seq_iobj = [filtered_names[i] for i in sorted_indices]
        sorted_centers = [filtered_centers[i] for i in sorted_indices]
        self.depths = [self.depths[i] for i in sorted_indices]

        self.distance_ratios = []
        for i in range(len(sorted_centers) - 1):
            distance = abs(sorted_centers[i + 1] - sorted_centers[i])
            self.distance_ratios.append(distance)
        if self.distance_ratios and sum(self.distance_ratios) > 0:
            total_distance = sum(self.distance_ratios)
            self.distance_ratios = [d / total_distance for d in self.distance_ratios]
        else:
            self.distance_ratios = [1.0] * (len(sorted_centers) - 1)

        # print("Detected objects:", self.seq_iobj)
        # print("Distance ratios:", self.distance_ratios)
        # print("Depths:", self.depths)

        for name in self.seq_iobj:
            for obj in self.obj_detect:
                if obj[0] == name:
                    obj[2] += 1

        for obj in self.obj_detect:
            name, obj_id, count, is_goal = obj
            graph_count = sum(1 for o in self.objects if o['name'] == name)
            if count > graph_count:
                logging.warning(f"Excess detections for {name}: detected {count}, graph has {graph_count}")
                obj[2] = graph_count
        # logging.info(f"Updated OBJ_DETECT: {self.obj_detect}")
        return len(self.seq_iobj) >= 2

    def build_local_graph(self, fov=60, image_width=640):
        """Build local graph from detected objects using pixel positions and depths."""
        self.local_graph = []
        fov_rad = np.deg2rad(fov)
        focal_length = (image_width / 2) / np.tan(fov_rad / 2)

        for i, (name, depth) in enumerate(zip(self.seq_iobj, self.depths)):
            if depth == 0:
                logging.warning(f"Invalid depth for object {name}, skipping")
                continue
            pixel_x = (i + 0.5) * (image_width / len(self.seq_iobj)) - (image_width / 2)
            angle = np.arctan2(pixel_x, focal_length)
            x = depth * np.sin(angle)
            y = depth * np.cos(angle)
            obj_id = str(uuid.uuid4())
            self.local_graph.append({
                'id': obj_id,
                'name': name,
                'bbox_center': [float(x), float(y), 0.0],
                'description': 'obstacle'
            })
        # logging.info(f"Local graph: {self.local_graph}")
        return len(self.local_graph) >= 2

    def align_graphs(self):
        import time
        start_time = time.perf_counter()
        """Align local graph with scene graph using ICP."""
        detected_names = set(obj['name'] for obj in self.local_graph)
        graph_objects = [obj for obj in self.objects if obj['name'] in detected_names]
        if len(graph_objects) < len(self.local_graph):
            logging.error("Not enough matching objects in scene graph")
            return None

        local_points = np.array([obj['bbox_center'][:2] for obj in self.local_graph])
        graph_points = np.array([obj['bbox_center'][:2] for obj in graph_objects])
        local_names = [obj['name'] for obj in self.local_graph]
        graph_names = [obj['name'] for obj in graph_objects]
        graph_ids = [str(obj['id']) for obj in graph_objects]

        def compute_error(params, src, dst, src_names, dst_names):
            tx, ty, theta = params
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            transformed = (R @ src.T).T + np.array([tx, ty])
            error = 0
            for i, (p1, n1) in enumerate(zip(transformed, src_names)):
                # print("n1 ", n1, p1)
                min_dist = float('inf')
                for p2, n2 in zip(dst, dst_names):
                    if n1 == n2:
                        # print("n2 ", n2, p2)
                        dist = np.linalg.norm(p1 - p2)
                        min_dist = min(min_dist, dist)
                # print("error ", n1, n2, min_dist)
                error += min_dist
            return error

        # Многократная минимизация
        N_TRIALS = 10  # Количество случайных запусков
        best_error = float('inf')
        best_params = None

        for _ in range(1):
            # Случайное начальное приближение
            tx_init = 2.7#np.random.uniform(0, 8)  # Диапазон по x и y, зависит от сцены
            ty_init = 1.4#np.random.uniform(0, 6)
            theta_init = np.random.uniform(-np.pi, np.pi)  # Диапазон углов
            initial_guess = [tx_init, ty_init, theta_init]

            # Минимизация
            result = minimize(
                compute_error,
                initial_guess,
                args=(local_points, graph_points, local_names, graph_names),
                method='Powell'
            )

            if result.success:
                current_error = result.fun  # Итоговая ошибка
                if current_error < best_error:
                    best_error = current_error
                    best_params = result.x

        if best_params is None:
            logging.error("Graph alignment failed after multiple trials")
            return None

        # Извлечение лучших параметров
        tx, ty, theta = best_params
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        matched_set = []
        transformed_points = (R @ local_points.T).T + np.array([tx, ty])

        for i, (p, n) in enumerate(zip(transformed_points, local_names)):
            min_dist = float('inf')
            best_id = None
            for j, (p2, n2, obj_id) in enumerate(zip(graph_points, graph_names, graph_ids)):
                if n == n2:
                    dist = np.linalg.norm(p - p2)
                    if dist < min_dist:
                        min_dist = dist
                        best_id = obj_id
            if best_id:
                matched_set.append(best_id)

        if len(matched_set) < len(self.local_graph):
            logging.error("Incomplete graph matching")
            return None

        robot_pos_local = np.array([0, 0])
        robot_pos_global = (R @ robot_pos_local) + np.array([tx, ty])
        self.robot_position = robot_pos_global
        self.closest_detected_set = matched_set
        logging.info(f"Aligned graph, robot position: {self.robot_position}, matched set: {matched_set}")
        # End timing
        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print("______________________________________________________TIME:", elapsed_time)
        return matched_set

    def compute_convex_hulls(self, convex_sets):
        """Compute convex hulls for matched sets."""
        hulls = []
        for convex_set in convex_sets:
            points = []
            for obj_id in convex_set:
                for obj in self.objects:
                    if str(obj['id']) == obj_id:
                        points.append([float(ele) for ele in obj['bbox_center'][:2]])
            points = np.array(points)
            if len(points) < 3:
                hull = {'points': points, 'area': 0, 'centroid': np.mean(points, axis=0)}
            else:
                try:
                    hull_obj = ConvexHull(points)
                    hull_points = points[hull_obj.vertices]
                    area = hull_obj.volume
                    centroid = np.mean(hull_points, axis=0)
                    hull = {'points': hull_points, 'area': area, 'centroid': centroid}
                except:
                    hull = {'points': points, 'area': 0, 'centroid': np.mean(points, axis=0)}
            hulls.append((convex_set, hull))
        return hulls

    def evaluate_by_circular_projections(self, hulls):
        """Evaluate observer position using depth and order constraints."""
        best_score = float('inf')
        best_result = None

        for convex_set, hull in hulls:
            ids = [str(obj_id) for obj_id in convex_set]
            id_to_point = {str(obj['id']): np.array(obj['bbox_center'][:2]) for obj in self.objects if str(obj['id']) in ids}
            id_to_name = {str(obj['id']): obj['name'] for obj in self.objects if str(obj['id']) in ids}
            points = list(id_to_point.values())
            centroid = hull['centroid']
            radius = max(np.linalg.norm(p - centroid) for p in points) if len(points) >= 2 else 0

            # Find closest object and its depth
            closest_id = ids[0]
            min_dist = float('inf')
            for i, (name, depth) in enumerate(zip(self.seq_iobj, self.depths)):
                if name == id_to_name[closest_id] and depth > 0:
                    min_dist = depth
                    break

            for angle in np.linspace(0, 2 * np.pi, 18, endpoint=False):
                direction = np.array([np.cos(angle), np.sin(angle)])
                normal = np.array([-direction[1], direction[0]])
                projections = []
                for obj_id in ids:
                    point = id_to_point[obj_id]
                    proj = np.dot(point - centroid, normal)
                    projections.append((proj, id_to_name[obj_id]))
                projections.sort()
                order = [name for _, name in projections]

                mismatch = sum(1 for a, b in zip(order, self.seq_iobj) if a != b)
                projected_dists = [abs(projections[i + 1][0] - projections[i][0]) for i in range(len(projections) - 1)]
                if projected_dists and sum(projected_dists) > 0:
                    total_dist = sum(projected_dists)
                    normalized_proj_dists = [d / total_dist for d in projected_dists]
                else:
                    normalized_proj_dists = [1.0] * len(projected_dists)

                ratio_error = 0
                if len(normalized_proj_dists) == len(self.distance_ratios):
                    ratio_error = sum((a - b) ** 2 for a, b in zip(normalized_proj_dists, self.distance_ratios)) / len(normalized_proj_dists)
                else:
                    ratio_error = float('inf')

                # Move plane to closest object
                closest_point = id_to_point[closest_id]
                plane_distance = np.dot(closest_point - centroid, direction)
                adjusted_distance = plane_distance + min_dist
                observer = centroid + direction * adjusted_distance
                score = mismatch + ratio_error + mismatch * 100

                # logging.info(f"[EVAL] Set {ids}, angle={angle:.2f}, order={order}, mismatch={mismatch}, score={score:.3f}")
                if score < best_score:
                    best_score = score
                    best_result = {
                        'observer': observer,
                        'orientation': -direction,
                        'order': order,
                        'set': convex_set
                    }

        if best_result:
            self.closest_detected_set = best_result['set']
            # self.robot_position = best_result['observer']
            self.robot_orientation = best_result['orientation']
            # logging.info(f"[BEST MATCH] order={best_result['order']}, observer={self.robot_position}, orientation={self.robot_orientation}")
            return True
        return False

    def update_scene_with_robot(self):
        """Place robot on grid."""
        centers = [obj['bbox_center'][:2] for obj in self.objects]
        centers_array = np.array(centers)
        min_coords = np.min(centers_array, axis=0)
        print("min ", min_coords)
        max_coords = np.max(centers_array, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1
        # print("robot position is ", self.robot_position)
        x, y = self.robot_position
        scaled_x = (x - min_coords[0]) / range_coords[0] * (self.inner_grid_size - 1) + 1
        scaled_y = (y - min_coords[1]) / range_coords[1] * (self.inner_grid_size - 1) + 1
        grid_x = int(round(scaled_x))
        grid_y = int(round(self.grid_size - 1 - scaled_y))
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))
        self.scene[grid_y][grid_x] = ['R']
        logging.info(f"Robot placed at {self.robot_position}, grid: ({grid_y}, {grid_x})")
        return True

    def generate_output_grids(self):
        """Generate output grid with format 'type(name_id)'."""
        grid = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        id_to_obj = {str(obj['id']): obj for obj in self.objects}
        detected_ids = set(str(obj_id) for obj_id in self.closest_detected_set) if self.closest_detected_set else set()

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.scene[i][j] == ['R']:
                    grid[i][j] = 'R'
                elif self.scene[i][j]:
                    entries = []
                    for obj_id in self.scene[i][j]:
                        obj = id_to_obj.get(obj_id)
                        if obj:
                            name_short = obj['name'][:3]
                            obj_type = 'g' if obj['description'] == 'goal' else 'o'
                            entry = f"{obj_type}({name_short}_{obj_id})"
                            if obj_id in detected_ids:
                                entry += '*'
                            entries.append(entry)
                    grid[i][j] = ','.join(entries)
        return grid

    def plot_scene(self, grid, filename):
        """Plot the scene with robot and ground truth positions."""
        import matplotlib.pyplot as plt
        import numpy as np
        import json
        import os
        import logging
        from scipy.spatial.transform import Rotation

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Robot Localization Scene ({filename})")
        rows, cols = len(grid), len(grid[0])
        
        # Draw grid lines
        for i in range(rows + 1):
            ax.axhline(i, color='black', linewidth=1)
        for j in range(cols + 1):
            ax.axvline(j, color='black', linewidth=1)
        
        # Find robot position on grid
        robot_pos = None
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 'R':
                    robot_pos = (j + 0.5, rows - i - 0.5)
                    break
            if robot_pos:
                break
        
        # Plot objects
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] and grid[i][j] != 'R':
                    entries = grid[i][j].split(',')
                    for entry in entries:
                        if not entry:
                            continue
                        is_bold = entry.endswith('*')
                        display_text = entry.rstrip('*')
                        fontweight = 'bold' if is_bold else 'normal'
                        ax.text(
                            j + 0.5, rows - i - 0.5, display_text,
                            ha='center', va='center', fontsize=10,
                            fontweight=fontweight,
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
                        )
        
        has_legend_items = False
        
        # Plot computed robot position and orientation
        if robot_pos and self.robot_orientation is not None:
            ax.scatter(*robot_pos, color='red', s=100, label='Robot (R)')
            dx = self.robot_orientation[0]
            dy = self.robot_orientation[1]
            # Normalize orientation vector for consistent arrow length
            norm = np.linalg.norm([dx, dy])
            if norm > 0:
                dx, dy = dx / norm, dy / norm
            ax.arrow(robot_pos[0], robot_pos[1], dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue', label='Robot Orientation')
            has_legend_items = True
        
        # Load and plot ground truth from camera_poses.json
        gt_path = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/dataset_info/camera_poses.json"
        try:
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            frame_key = f"frame{filename}.jpg"
            print("frame_key is ", frame_key)
            if frame_key in gt_data:
                gt_pose = gt_data[frame_key]
                gt_position = np.array(gt_pose['position'][:2])  # Take x, y
                gt_quaternion = np.array(gt_pose['orientation'])  # [x, y, z, w]
                print("gt pose is ", gt_position)
                
                # Transform ground truth position to grid coordinates
                centers = [obj['bbox_center'][:2] for obj in self.objects]
                centers_array = np.array(centers)
                min_coords = np.min(centers_array, axis=0)
                print("min ", min_coords)
                max_coords = np.max(centers_array, axis=0)
                range_coords = max_coords - min_coords
                range_coords[range_coords == 0] = 1
                
                x, y = gt_position
                scaled_x = (x - min_coords[0]) / range_coords[0] * (self.inner_grid_size - 1) + 1
                scaled_y = (y - min_coords[1]) / range_coords[1] * (self.inner_grid_size - 1) + 1
                print("_______________log ", scaled_x, scaled_y)
                grid_x = int(round(scaled_x))
                grid_y = int(round(self.grid_size - 1 - scaled_y))
                # grid_x = max(0, min(self.grid_size - 1, grid_x))
                # grid_y = max(0, min(self.grid_size - 1, grid_y))
                # print("_______________log ", scaled_x, scaled_y)
                robot_true_pos = (grid_x + 0.5, grid_y + 0.5)
                print("_______________log ", robot_true_pos, robot_pos)
                # Convert quaternion to 2D orientation vector
                # Assuming rotation is primarily around Z-axis (yaw)
                rot = Rotation.from_quat(gt_quaternion)  # quaternion in [x, y, z, w]
                # Get yaw angle (rotation around Z-axis)
                euler = rot.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
                yaw = euler[2]  # Take yaw (rotation in XY plane)
                # Convert yaw to 2D direction vector
                gt_dx = np.cos(yaw)
                gt_dy = np.sin(yaw)
                
                # Plot ground truth position and orientation
                ax.scatter(robot_true_pos[0], robot_true_pos[1], color='green', s=100, label='Ground Truth (GT)', marker='o')
                ax.arrow(robot_true_pos[0], robot_true_pos[1], gt_dx, gt_dy, head_width=0.2, head_length=0.2, fc='green', ec='green', label='GT Orientation')
                has_legend_items = True
                logging.info(f"Plotted ground truth for {frame_key}: position={gt_position}, quaternion={gt_quaternion}, yaw={yaw:.3f}, direction=[{gt_dx:.3f}, {gt_dy:.3f}]")
            else:
                logging.warning(f"No ground truth data for {frame_key}")
        except Exception as e:
            logging.error(f"Failed to load or plot ground truth: {str(e)}")
        
        # Plot coordinate axes
        ax.arrow(0, 0, 1, 0, head_width=0.5, head_length=1, fc='green', ec='green', label='X-axis')
        ax.arrow(0, 0, 0, 1, head_width=0.5, head_length=1, fc='red', ec='red', label='Y-axis')
        has_legend_items = True
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(range(cols + 1))
        ax.set_yticks(range(rows + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        
        if has_legend_items:
            ax.legend()
        
        plt.savefig(os.path.join(self.output_dir, f'grid_{filename}.png'))
        plt.close()
        logging.info(f"Saved plot: grid_{filename}.png")

    def run(self, rgb_path, depth_path, i_im):
        """Execute localization pipeline for one image pair."""
        self.frame_num = i_im
        self.build_scene_grid()
        if not self.detect_objects(rgb_path, depth_path):
            logging.error("Less than 2 objects detected")
            return None
        if not self.build_local_graph():
            logging.error("Failed to build local graph")
            return None
        local_grid = self.build_local_grid()
        self.plot_local_graph(local_grid, i_im)

        convex_set = self.align_graphs()
        if not convex_set:
            logging.error("Graph alignment failed")
            return None
        hulls = self.compute_convex_hulls([convex_set])
        if not self.evaluate_by_circular_projections(hulls):
            logging.error("No valid observer position found")
            return None
        self.update_scene_with_robot()
        grid = self.generate_output_grids()
        self.plot_scene(grid, i_im)

        # Verify localization results
        from verify_localization import verify_localization
        ground_truth_path = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/dataset_info/camera_poses.json"
        verify_localization(ground_truth_path, self.output_dir, i_im, self.robot_position, self.robot_orientation)

        return grid

def main(rgb_dir, depth_dir, graph_path, output_dir):
    """Main function to process all image pairs."""
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    locator = RobotLocalization(rgb_dir, depth_dir, graph_path, output_dir)
    if not locator.get_image_pairs():
        logging.error("No valid image pairs found")
        return
    for i, (rgb_path, depth_path) in enumerate(zip(locator.rgb_images, locator.depth_images)):
        print(f"_______________________________Image pair {i}")
        # Извлекаем номер кадра из имени RGB-файла
        frame_num = re.search(r'frame(\d+)\.jpg', os.path.basename(rgb_path)).group(1)
        try:
            locator.run(rgb_path, depth_path, frame_num)
        except Exception as e:
            logging.error(f"Error processing image pair {i}: {str(e)}")
            continue


if __name__ == "__main__":
    rgb_dir = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/rgb_dataset"
    depth_dir = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/d_dataset"
    graph_path = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/test_graph.json"
    output_dir = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/output_yolo"

    main(rgb_dir, depth_dir, graph_path, output_dir)