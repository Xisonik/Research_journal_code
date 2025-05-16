import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def get_simple_embedding_from_json(json_file, model_name='all-MiniLM-L6-v2', max_objects=10):
    """
    Генерирует эмбеддинги для всех объектов из JSON-файла с фиксированной размерностью.
    
    Args:
        json_file (str): Путь к JSON-файлу с описанием сцены.
        model_name (str): Название модели sentence-transformers для генерации текстовых эмбеддингов.
        max_objects (int): Максимальное количество объектов, которое будет включено в итоговый тензор.
    
    Returns:
        torch.Tensor: Тензор формы (max_objects, 390), где каждый объект представлен отдельно.
                     Если объектов меньше max_objects, оставшиеся места заполняются нулями.
                     Если объектов больше max_objects, лишние обрезаются.
    """
    # Загружаем модель sentence-transformers
    model = SentenceTransformer(model_name)

    # Читаем JSON-файл
    with open(json_file, "r") as f:
        data = json.load(f)

    object_features = []

    # Обрабатываем каждый объект в JSON
    for obj_data in data:
        # Извлекаем числовые характеристики
        bbox_extent = torch.tensor(obj_data["bbox_extent"], dtype=torch.float32)  # 3 dimensions
        bbox_center = torch.tensor(obj_data["bbox_center"], dtype=torch.float32)  # 3 dimensions

        # Извлекаем текстовое описание и генерируем эмбеддинг
        description = obj_data["description"]
        text_embedding = model.encode(description, convert_to_tensor=True)  # 384 dimensions (для all-MiniLM-L6-v2)

        # Объединяем числовые характеристики и текстовый эмбеддинг
        features = torch.cat([bbox_extent, bbox_center, text_embedding], dim=0)  # 3 + 3 + 384 = 390 dimensions
        object_features.append(features)

    # Если объектов нет, возвращаем тензор с нулями
    if not object_features:
        return torch.zeros((max_objects, 390), dtype=torch.float32)

    # Преобразуем список в тензор
    object_tensor = torch.stack(object_features)  # Shape: (num_objects, 390)

    # Фиксируем размерность
    num_objects = object_tensor.shape[0]
    feature_dim = object_tensor.shape[1]  # 390

    # Создаем итоговый тензор фиксированной формы (max_objects, 390)
    fixed_tensor = torch.zeros((max_objects, feature_dim), dtype=torch.float32)

    if num_objects <= max_objects:
        # Если объектов меньше или равно max_objects, заполняем тензор и оставляем нули в конце
        fixed_tensor[:num_objects, :] = object_tensor
    else:
        # Если объектов больше max_objects, обрезаем лишние
        fixed_tensor = object_tensor[:max_objects, :]

    return fixed_tensor

# Пример использования
json_file = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/gjst/obstacles_and_goal_3_1.json"  # Укажите путь к вашему JSON-файлу
scene_embeddings = get_simple_embedding_from_json(json_file, max_objects=10)
print("Scene embeddings shape:", scene_embeddings.shape)
print("Scene embeddings:", scene_embeddings)