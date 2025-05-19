import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

class DepthCameraValidator:
    """Модуль для проверки корректности работы камеры глубины."""
    
    def __init__(self, depth_dir, rgb_dir, output_dir, max_room_depth=8.0, scale_factor=1000.0):
        """
        Инициализация валидатора камеры глубины.
        
        Args:
            depth_dir (str): Путь к папке с глубинными изображениями (depthXXXXXX.png).
            rgb_dir (str): Путь к папке с RGB-изображениями (frameXXXXXX.jpg).
            output_dir (str): Путь к папке для сохранения результатов анализа.
            max_room_depth (float): Максимальная глубина комнаты в метрах (по умолчанию 8.0).
            scale_factor (float): Коэффициент масштабирования глубины (по умолчанию 1000, предполагает мм -> м).
        """
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir
        self.output_dir = output_dir
        self.max_room_depth = max_room_depth
        self.scale_factor = scale_factor
        
        # Создание папки вывода
        os.makedirs(output_dir, exist_ok=True)
        
        # Настройка логирования
        log_file = os.path.join(output_dir, f'depth_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        logging.info(f"Initialized DepthCameraValidator: depth_dir={depth_dir}, rgb_dir={rgb_dir}, output_dir={output_dir}")

    def get_image_pairs(self):
        """Получение пар RGB- и глубинных изображений."""
        rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.startswith('frame') and f.endswith('.jpg')])
        depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.startswith('depth') and f.endswith('.png')])
        
        pairs = []
        for rgb_file in rgb_files:
            frame_num = rgb_file.replace('frame', '').replace('.jpg', '')
            depth_file = f'depth{frame_num}.png'
            if depth_file in depth_files:
                pairs.append({
                    'rgb_path': os.path.join(self.rgb_dir, rgb_file),
                    'depth_path': os.path.join(self.depth_dir, depth_file),
                    'frame_num': frame_num
                })
            else:
                logging.warning(f"No depth image found for {rgb_file}")
        
        logging.info(f"Found {len(pairs)} valid image pairs")
        return pairs

    def validate_depth_image(self, depth_path, rgb_path, frame_num):
        """Проверка одного глубинного изображения."""
        # Загрузка изображений
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.imread(rgb_path)
        
        if depth_img is None:
            logging.error(f"Failed to load depth image: {depth_path}")
            return False
        if rgb_img is None:
            logging.warning(f"Failed to load RGB image: {rgb_path}")
            rgb_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        
        # Преобразование глубины
        depth_img = depth_img.astype(np.float32)
        raw_min, raw_max = np.min(depth_img), np.max(depth_img)
        logging.info(f"Frame {frame_num}: Raw depth range: min={raw_min:.3f}, max={raw_max:.3f}")
        
        # Проверка отрицательных значений
        negative_count = np.sum(depth_img < 0)
        if negative_count > 0:
            logging.warning(f"Frame {frame_num}: Found {negative_count} negative depth values")
            depth_img = np.abs(depth_img)  # Убираем отрицательные значения
        
        # Проверка нулевых значений
        zero_count = np.sum(depth_img == 0)
        if zero_count > 0:
            logging.warning(f"Frame {frame_num}: Found {zero_count} zero depth values ({100 * zero_count / depth_img.size:.2f}%)")
        
        # Масштабирование в метры
        depth_scaled = depth_img / self.scale_factor
        depth_scaled[depth_scaled == 0] = np.nan  # Заменяем 0 на NaN
        depth_scaled = np.clip(depth_scaled, 0.1, self.max_room_depth)  # Ограничение [0.1, max_room_depth]
        
        # Анализ масштабированных значений
        valid_depths = depth_scaled[~np.isnan(depth_scaled)]
        if valid_depths.size == 0:
            logging.error(f"Frame {frame_num}: No valid depth values after scaling")
            return False
        
        depth_min, depth_max = np.nanmin(depth_scaled), np.nanmax(depth_scaled)
        depth_mean = np.nanmean(depth_scaled)
        depth_median = np.nanmedian(depth_scaled)
        logging.info(f"Frame {frame_num}: Scaled depth (m): min={depth_min:.3f}, max={depth_max:.3f}, mean={depth_mean:.3f}, median={depth_median:.3f}")
        
        # Проверка аномалий
        if depth_max > self.max_room_depth:
            logging.warning(f"Frame {frame_num}: Max depth ({depth_max:.3f} m) exceeds room limit ({self.max_room_depth} m)")
        if depth_min < 0.1:
            logging.warning(f"Frame {frame_num}: Min depth ({depth_min:.3f} m) too small, possible noise")
        
        point_coords = (50,50)
        if point_coords is not None:
            x, y = point_coords
            if 0 <= x < depth_scaled.shape[1] and 0 <= y < depth_scaled.shape[0]:
                point_depth = depth_scaled[y, x]
                if not np.isnan(point_depth):
                    logging.info(f"Frame {frame_num}: Depth at point ({x}, {y}) = {point_depth:.3f} m")
                else:
                    logging.warning(f"Frame {frame_num}: No valid depth at point ({x}, {y})")
            else:
                logging.warning(f"Frame {frame_num}: Point ({x}, {y}) out of image bounds ({depth_scaled.shape[1]}x{depth_scaled.shape[0]})")

        # Визуализация
        self.visualize_depth(depth_scaled, rgb_img, frame_num)
        
        return True

    def visualize_depth(self, depth_scaled, rgb_img, frame_num):
        """Визуализация глубинного изображения и гистограммы."""
        # Нормализация для отображения
        depth_display = depth_scaled.copy()
        depth_display[np.isnan(depth_display)] = 0
        depth_display = (depth_display / self.max_room_depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        
        # Создание фигуры
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Depth Validation: Frame {frame_num}")
        
        # RGB-изображение
        axes[0, 0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("RGB Image")
        axes[0, 0].axis('off')
        
        # Глубинное изображение (сырое)
        axes[0, 1].imshow(depth_scaled, cmap='jet', vmin=0, vmax=self.max_room_depth)
        axes[0, 1].set_title("Depth Image (m)")
        axes[0, 1].axis('off')
        plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
        
        # Глубинное изображение (цветное)
        axes[1, 0].imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Depth Image (Colored)")
        axes[1, 0].axis('off')
        
        # Гистограмма
        valid_depths = depth_scaled[~np.isnan(depth_scaled)]
        axes[1, 1].hist(valid_depths, bins=50, range=(0, self.max_room_depth), density=True)
        axes[1, 1].set_title("Depth Histogram")
        axes[1, 1].set_xlabel("Depth (m)")
        axes[1, 1].set_ylabel("Density")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(self.output_dir, f'depth_analysis_{frame_num}.png')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved depth visualization: {output_path}")

    def run(self):
        """Запуск проверки для всех пар изображений."""
        pairs = self.get_image_pairs()
        if not pairs:
            logging.error("No valid image pairs found")
            return
        
        for pair in pairs:
            logging.info(f"Processing frame {pair['frame_num']}")
            try:
                self.validate_depth_image(
                    pair['depth_path'],
                    pair['rgb_path'],
                    pair['frame_num']
                )
            except Exception as e:
                logging.error(f"Error processing frame {pair['frame_num']}: {str(e)}")

def main():
    """Точка входа для запуска валидатора."""
    rgb_dir = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/rgb_dataset"
    depth_dir = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/d_dataset"
    graph_path = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/test_graph.json"
    output_dir = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/local_module/output_test"
    
    # Настройка параметров (уточните масштаб вашей камеры)
    max_room_depth = 8.0  # Максимальная глубина комнаты в метрах
    scale_factor = 10000.0  # Предполагаемый масштаб: миллиметры -> метры
    
    validator = DepthCameraValidator(depth_dir, rgb_dir, output_dir, max_room_depth, scale_factor)
    validator.run()

if __name__ == "__main__":
    main()