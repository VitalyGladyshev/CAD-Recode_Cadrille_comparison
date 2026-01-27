#!/usr/bin/env python3
"""
Тестирование модели Cadrille с генерацией CadQuery кодов.

Модуль предоставляет функциональность для запуска инференса модели Cadrille
на различных датасетах. Поддерживает работу с текстовыми описаниями,
точечными облаками и изображениями 3D моделей.

Основные компоненты:
    - Функция run(): основной интерфейс для запуска тестирования модели
    - Поддержка различных режимов: 'text', 'pc', 'img'
    - Автоматическая обработка батчей и сохранение результатов

Используемые библиотеки:
    - torch: для работы с моделями и тензорами
    - transformers: для загрузки процессора и работы с токенами
    - cadrille: модель Cadrille для генерации кодов
    - dataset: классы датасетов для загрузки данных

Пример использования:
    >>> from test import run
    >>> run(
    ...     data_path='/workspace/data',
    ...     split='deepcad_test_mesh',
    ...     mode='pc',
    ...     checkpoint_path='/workspace/models/cadrille-rl',
    ...     py_path='/workspace/results/cadrille',
    ...     device='cuda',
    ...     max_samples=50
    ... )
"""
import os
from tqdm import tqdm
from functools import partial
from argparse import ArgumentParser
import logging

import torch
from transformers import AutoProcessor
from torch.utils.data import DataLoader, ConcatDataset

from cadrille import Cadrille, collate
from dataset import Text2CADDataset, CadRecodeDataset

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run(data_path: str, split: str, mode: str, checkpoint_path: str, py_path: str, 
        device: str = 'cuda', max_samples: int = None) -> None:
    """
    Запускает тестирование модели Cadrille на указанном датасете.
    
    Генерирует CadQuery коды для 3D моделей из датасета и сохраняет их
    в указанную директорию. Поддерживает режимы работы с текстом, точечными
    облаками и изображениями.
    
    Args:
        data_path (str): Путь к корневой директории с данными.
            Должна содержать поддиректории с датасетами.
        split (str): Название датасета для тестирования.
            Например, 'deepcad_test_mesh' или 'fusion360_test_mesh'.
        mode (str): Режим работы модели. Допустимые значения:
            - 'text': текстовый режим (использует Text2CADDataset)
            - 'pc': режим с точечными облаками (использует CadRecodeDataset)
            - 'img': режим с изображениями (использует CadRecodeDataset)
        checkpoint_path (str): Путь к чекпоинту модели Cadrille.
            Может быть локальным путем или HuggingFace моделью.
        py_path (str): Директория для сохранения сгенерированных CadQuery файлов.
            Должна быть пустой перед запуском функции.
        device (str, optional): Устройство для вычислений ('cuda' или 'cpu').
            По умолчанию 'cuda'.
        max_samples (int, optional): Максимальное количество образцов для обработки.
            Если None, обрабатываются все образцы из датасета.
            По умолчанию None.
    
    Raises:
        AssertionError: Если директория py_path не пуста перед запуском.
        FileNotFoundError: Если checkpoint_path не существует.
        ValueError: Если mode имеет недопустимое значение.
        RuntimeError: Если не удалось загрузить модель или обработать данные.
    
    Note:
        - Директория py_path должна быть пустой перед запуском функции.
          После выполнения в ней будут сохранены .py файлы с CadQuery кодом.
        - Для режима 'text' используется batch_size=32, для 'pc' и 'img' - batch_size=8.
        - Используется num_workers=20 для параллельной загрузки данных.
        - Модель автоматически загружается на указанное устройство (GPU/CPU).
    
    Example:
        >>> run(
        ...     data_path='/workspace/data',
        ...     split='deepcad_test_mesh',
        ...     mode='pc',
        ...     checkpoint_path='/workspace/models/cadrille-rl',
        ...     py_path='/workspace/results/cadrille',
        ...     device='cuda',
        ...     max_samples=50
        ... )
    """
    # Проверяем, что директория для результатов пуста (избегаем перезаписи)
    os.makedirs(py_path, exist_ok=True)
    assert len(os.listdir(py_path)) == 0

    # Загрузка модели Cadrille с оптимизациями для GPU
    model = Cadrille.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device_map=device if device == 'cuda' else None,
        attn_implementation='flash_attention_2' if device == 'cuda' else None)

    # Загрузка процессора для обработки входных данных
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct', 
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28,
        padding_side='left')

    # Выбор датасета и размера батча в зависимости от режима
    if mode == 'text':
        dataset = Text2CADDataset(
            root_dir=os.path.join(data_path, 'text2cad'),
            split='test')
        batch_size = 32
    else:  # mode in ('pc', 'img')
        dataset = CadRecodeDataset(
            root_dir=data_path,
            split=split,
            n_points=256,
            normalize_std_pc=100,
            noise_scale_pc=None,
            img_size=128,
            normalize_std_img=200,
            noise_scale_img=-1,
            num_imgs=4,
            mode=mode,
            n_samples=max_samples)
        batch_size = 8

    n_samples = 1
    counter = 0
    dataset_len = len(dataset)
    
    # Используем ConcatDataset только если n_samples > 1 (для множественных генераций)
    if n_samples > 1:
        concat_dataset = ConcatDataset([dataset] * n_samples)
    else:
        concat_dataset = dataset
    
    # Создание DataLoader с параллельной загрузкой данных
    dataloader = DataLoader(
        dataset=concat_dataset,
        batch_size=batch_size,
        num_workers=20,  # Используем все 20 ядер процессора для загрузки данных
        collate_fn=partial(collate, processor=processor, n_points=256, eval=True))

    # Обработка батчей и генерация CadQuery кодов
    for batch in tqdm(dataloader):
        # Генерация токенов модели
        generated_ids = model.generate(
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device),
            point_clouds=batch['point_clouds'].to(model.device),
            is_pc=batch['is_pc'].to(model.device),
            is_img=batch['is_img'].to(model.device),
            pixel_values_videos=batch['pixel_values_videos'].to(model.device) if batch.get('pixel_values_videos', None) is not None else None,
            video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get('video_grid_thw', None) is not None else None,
            max_new_tokens=768)
        
        # Извлечение только сгенерированных токенов (без входных)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
        ]
        
        # Декодирование токенов в строки CadQuery кода
        py_strings = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Сохранение сгенерированных кодов в файлы
        for stem, py_string in zip(batch['file_name'], py_strings):
            # Вычисляем индекс сэмпла для множественных генераций
            sample_idx = counter // dataset_len if dataset_len > 0 else 0
            file_name = f'{stem}+{sample_idx}.py'
            with open(os.path.join(py_path, file_name), 'w') as f:
                f.write(py_string)
            counter += 1


def main():
    """
    Главная функция для запуска скрипта из командной строки.
    
    Парсит аргументы командной строки и запускает тестирование модели Cadrille.
    Выполняет проверку доступности GPU и выводит информацию о системе.
    """
    parser = ArgumentParser(description='Тестирование модели Cadrille')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Путь к корневой директории с данными')
    parser.add_argument('--split', type=str, default='deepcad_test_mesh',
                        help='Название датасета для тестирования')
    parser.add_argument('--mode', type=str, default='pc',
                        choices=['text', 'pc', 'img'],
                        help='Режим работы модели: text, pc или img')
    parser.add_argument('--checkpoint-path', type=str, default='maksimko123/cadrille',
                        help='Путь к чекпоинту модели Cadrille')
    parser.add_argument('--py-path', type=str, default='./work_dirs/tmp_py',
                        help='Директория для сохранения сгенерированных CadQuery файлов')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Устройство для вычислений (cuda/cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Максимальное количество объектов для обработки')
    args = parser.parse_args()

    # Проверка доступности GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA недоступна, переключаемся на CPU")
        args.device = 'cpu'
    
    if torch.cuda.is_available():
        logger.info(f"GPU доступен: {torch.cuda.get_device_name(0)}")
        logger.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    run(
        args.data_path, 
        args.split, 
        args.mode, 
        args.checkpoint_path, 
        args.py_path, 
        args.device,
        args.max_samples)


if __name__ == '__main__':
    main()
