import json
import os
from pathlib import Path

# Путь к директории логов
d = Path().resolve()
general_path = str(d)
log = general_path + "/logs/"
paths_file = os.path.join(log, "all_paths.json")


def load_paths():
    """Загружает данные из JSON-файла и преобразует ключи в кортежи."""
    if not os.path.exists(paths_file):
        print(f"Файл {paths_file} не найден!")
        return None

    try:
        with open(paths_file, 'r') as f:
            loaded_paths = json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке файла {paths_file}: {e}")
        return None

    # Преобразуем строковые ключи обратно в кортежи
    all_paths = {}
    for config_key, targets in loaded_paths.items():
        k0, k1 = map(int, config_key.split('_'))
        config_tuple = (k0, k1)
        all_paths[config_tuple] = {}
        for target_str, nodes in targets.items():
            target = tuple(map(int, target_str.split(',')))
            all_paths[config_tuple][target] = {}
            for node_str, path in nodes.items():
                node = tuple(map(int, node_str.split(',')))
                all_paths[config_tuple][target][node] = [tuple(p) for p in path]
    
    return all_paths


def get_config_key(all_paths):
    """Запрашивает у пользователя два числа для выбора конфигурации."""
    config_keys = list(all_paths.keys())
    print(f"\nДоступные конфигурации препятствий (всего {len(config_keys)}):")
    print("Примеры:", config_keys[:5], "..." if len(config_keys) > 5 else "")
    print("Конфигурация задается двумя числами от 0 до 7 (например, '0 1' для (0, 1)).")

    while True:
        try:
            user_input = input("Введите два числа через пробел (k0 k1): ").strip().split()
            if len(user_input) != 2:
                print("Введите ровно два числа!")
                continue
            k0, k1 = map(int, user_input)
            if not (0 <= k0 <= 7 and 0 <= k1 <= 7):
                print("Числа должны быть в диапазоне от 0 до 7!")
                continue
            config_key = (k0, k1)
            if config_key not in all_paths:
                print(f"Конфигурация {config_key} не найдена в данных!")
                continue
            return config_key
        except ValueError:
            print("Введите два целых числа!")


def display_options(options, title):
    """Отображает список опций и возвращает выбор пользователя."""
    print(f"\n{title}:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        try:
            choice = int(input(f"Выберите номер (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print("Неверный номер, попробуйте снова.")
        except ValueError:
            print("Введите число!")


def get_path_info(all_paths):
    """Основная функция для навигации по JSON-файлу."""
    if all_paths is None:
        return

    # Шаг 1: Выбор конфигурации препятствий двумя числами
    selected_config = get_config_key(all_paths)

    # Шаг 2: Выбор цели
    targets = list(all_paths[selected_config].keys())
    print(f"\nКонфигурация {selected_config} имеет {len(targets)} целей.")
    selected_target = display_options(targets, "Доступные цели")

    # Шаг 3: Выбор стартовой позиции
    start_positions = list(all_paths[selected_config][selected_target].keys())
    print(f"\nЦель {selected_target} имеет {len(start_positions)} стартовых позиций.")
    selected_start = display_options(start_positions, "Доступные стартовые позиции")

    # Шаг 4: Вывод пути и метаданных
    path = all_paths[selected_config][selected_target][selected_start]
    path_length = len(path)

    print(f"\nВыбранная конфигурация: {selected_config}")
    print(f"Выбранная цель: {selected_target}")
    print(f"Выбранная стартовая позиция: {selected_start}")
    print(f"Путь: {path}")
    print(f"Длина пути: {path_length} узлов")


def main():
    """Точка входа скрипта."""
    print("Запуск навигации по файлу путей...")
    all_paths = load_paths()
    if all_paths:
        get_path_info(all_paths)
    else:
        print("Не удалось загрузить данные для навигации.")


if __name__ == "__main__":
    main()