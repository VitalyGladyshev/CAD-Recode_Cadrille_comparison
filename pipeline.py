#!/usr/bin/env python3
"""
Конвейер тестирования и оценки моделей CAD-Recode и Cadrille
Запускает полный цикл: тестирование -> оценка -> сравнение результатов
"""
import os
import subprocess
import sys
from pathlib import Path
from argparse import ArgumentParser
import time
import json
import torch
import gc
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_header(text: str) -> None:
    """
    Выводит форматированный заголовок в консоль.
    
    Создает визуально выделенный заголовок с текстом, центрированным в рамке
    из символов '='. Используется для улучшения читаемости вывода конвейера.
    
    Args:
        text (str): Текст заголовка для вывода.
            Будет центрирован в рамке шириной 80 символов.
    
    Example:
        >>> print_header("Запуск тестирования модели")
        ================================================================================
                                        Запуск тестирования модели
        ================================================================================
    """
    logger.info("\n" + "="*80)
    logger.info(f" {text} ".center(80, "="))
    logger.info("="*80)

def clear_gpu_memory() -> None:
    """
    Полностью очищает память GPU для освобождения ресурсов.
    
    Выполняет комплексную очистку памяти GPU: очищает кэш CUDA, запускает
    сборщик мусора Python, удаляет все CUDA тензоры из памяти и выводит
    информацию о свободной памяти. Используется между этапами конвейера
    для предотвращения переполнения памяти при работе с большими моделями.
    
    Процесс очистки:
        1. Очистка кэша CUDA через torch.cuda.empty_cache()
        2. Принудительная сборка мусора Python через gc.collect()
        3. Поиск и удаление всех CUDA тензоров в памяти
        4. Дополнительная очистка кэша CUDA
        5. Вывод информации о свободной памяти
    
    Note:
        Функция безопасна для вызова даже если GPU недоступна.
        Выводит информацию о свободной памяти только если GPU доступна.
    
    Example:
        >>> clear_gpu_memory()
        ✅ Память GPU очищена. Свободно: 18.50GB / 24.00GB
    """
    if torch.cuda.is_available():
        # Удаляем все закэшированные тензоры CUDA
        torch.cuda.empty_cache()
        
        # Принудительно запускаем сборщик мусора
        gc.collect()
        
        # Проверяем и удаляем все CUDA тензоры из памяти
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    del obj
            except Exception:
                pass
        
        # Дополнительная очистка кэша
        torch.cuda.empty_cache()
        
        # Выводим информацию о свободной памяти
        free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # в GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ Память GPU очищена. Свободно: {free_memory:.2f}GB / {total_memory:.2f}GB")
    else:
        logger.info("ℹ️ GPU не доступна для очистки")

def run_command(command: list, description: str) -> bool:
    """
    Запускает команду с выводом статуса и обработкой ошибок.
    
    Выполняет команду через subprocess.run() с перехватом вывода и обработкой
    ошибок. Используется для выполнения этапов конвейера (тестирование, оценка,
    сравнение результатов) с единообразной обработкой ошибок и логированием.
    
    Args:
        command (list): Список аргументов команды для выполнения.
            Первый элемент - исполняемый файл, остальные - аргументы.
            Например: ['python', 'test.py', '--arg', 'value']
        description (str): Описание команды для логирования.
            Используется в сообщениях об успехе/ошибке и заголовке.
    
    Returns:
        bool: True если команда выполнена успешно (returncode == 0),
            False в случае ошибки (subprocess.CalledProcessError).
    
    Note:
        Функция выводит заголовок с описанием, саму команду, время выполнения
        и результат (успех/ошибка). Использует check=True для автоматического
        выброса исключения при ошибке выполнения.
    
    Example:
        >>> success = run_command(
        ...     command=['python', 'test.py', '--data-path', '/workspace/data'],
        ...     description='Тестирование модели Cadrille'
        ... )
        >>> if not success:
        ...     print("Ошибка выполнения команды")
    """
    print_header(description)
    logger.info(f"Команда: {' '.join(command)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(command, check=True)
        elapsed = time.time() - start_time
        logger.info(f"\n✅ {description} успешно завершено за {elapsed:.2f} секунд")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ Ошибка при выполнении {description}: {e}")
        return False

def run_pipeline(data_path: str, models_dir: str, results_dir: str, 
                dataset: str = 'deepcad_test_mesh', n_samples: int = 1, 
                max_samples: int = 50, device: str = 'cuda') -> None:
    """
    Запускает полный конвейер тестирования и оценки моделей CAD.
    
    Выполняет последовательность этапов для тестирования и оценки моделей
    CAD-Recode и Cadrille на указанном датасете. Результаты сохраняются в
    структурированном виде для последующего анализа и сравнения.
    
    Процесс выполнения:
        1. Тестирование модели CAD-Recode (генерация CadQuery кодов)
        2. Оценка результатов CAD-Recode (вычисление метрик)
        3. Очистка памяти GPU
        4. Тестирование модели Cadrille (генерация CadQuery кодов)
        5. Оценка результатов Cadrille (вычисление метрик)
        6. Сравнение результатов обеих моделей
    
    Args:
        data_path (str): Путь к корневой директории с данными.
            Должна содержать поддиректорию с названием dataset.
        models_dir (str): Путь к директории с моделями.
            Должна содержать поддиректории 'cad-recode-v1.5' и 'cadrille-rl'.
        results_dir (str): Путь для сохранения результатов.
            Будет создана автоматически со следующей структурой:
            - cad_recode/{dataset}/: результаты тестирования CAD-Recode
            - cad_recode_eval/{dataset}/: результаты оценки CAD-Recode
            - cadrille/{dataset}/: результаты тестирования Cadrille
            - cadrille_eval/{dataset}/: результаты оценки Cadrille
            - comparison/{dataset}/: результаты сравнения моделей
        dataset (str, optional): Название датасета для тестирования.
            Должен быть поддиректорией в data_path со STL файлами.
            По умолчанию 'deepcad_test_mesh'.
        n_samples (int, optional): Количество генераций на каждый объект.
            Позволяет получить несколько вариантов кода для одного объекта.
            По умолчанию 1.
        max_samples (int, optional): Максимальное количество объектов для обработки.
            Если None, обрабатываются все найденные файлы.
            По умолчанию 50.
        device (str, optional): Устройство для вычислений ('cuda' или 'cpu').
            По умолчанию 'cuda'.
    
    Returns:
        None: Функция не возвращает значение, но создает файлы с результатами
            в указанной директории results_dir.
    
    Raises:
        FileNotFoundError: Если не найдены необходимые директории или модели
        subprocess.CalledProcessError: Если один из этапов завершился с ошибкой
    
    Note:
        Конвейер останавливается при ошибке на любом этапе. Память GPU очищается
        между тестированием CAD-Recode и Cadrille для предотвращения переполнения.
        Результаты сравнения сохраняются в JSON формате для последующего анализа.
    
    Example:
        >>> run_pipeline(
        ...     data_path='/workspace/data',
        ...     models_dir='/workspace/models',
        ...     results_dir='/workspace/results',
        ...     dataset='deepcad_test_mesh',
        ...     n_samples=1,
        ...     max_samples=50,
        ...     device='cuda'
        ... )
    """
    # Создание директории для результатов
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Подготовка путей
    cad_recode_model = str(Path(models_dir) / "cad-recode-v1.5")
    cadrille_model = str(Path(models_dir) / "cadrille-rl")
    
    # 1. Тестирование CAD-Recode
    cad_recode_output = results_dir / "cad_recode" / dataset
    cad_recode_output.mkdir(exist_ok=True, parents=True)
    
    success = run_command([
        "python", "test_cad_recode.py",
        "--data-path", data_path,
        "--split", dataset,
        "--model-path", cad_recode_model,
        "--output-dir", str(cad_recode_output),
        "--device", device,
        "--n-samples", str(n_samples),
        "--max-samples", str(max_samples)
    ], "Тестирование CAD-Recode")
    
    if not success:
        return
    
    # Оценка CAD-Recode
    cad_recode_eval_dir = results_dir / "cad_recode_eval" / dataset
    cad_recode_eval_dir.mkdir(exist_ok=True, parents=True)
    
    success = run_command([
        "python", "evaluate.py",
        "--gt-mesh-path", str(Path(data_path) / dataset),
        "--pred-py-path", str(cad_recode_output),
        "--pred-eval-path", str(cad_recode_eval_dir),
        "--n-points", "8192"
    ], "Оценка CAD-Recode")
    
    if not success:
        return

    clear_gpu_memory()  # Очищаем память перед загрузкой модели Cadrille
    
    # 2. Тестирование Cadrille
    cadrille_output = results_dir / "cadrille" / dataset
    cadrille_output.mkdir(exist_ok=True, parents=True)
    
    success = run_command([
        "python", "test.py",
        "--data-path", data_path,
        "--split", dataset,
        "--mode", "pc",
        "--checkpoint-path", cadrille_model,
        "--py-path", str(cadrille_output),
        "--device", device,
        "--max-samples", str(max_samples)
    ], "Тестирование Cadrille")
    
    if not success:
        return
    
    # Оценка Cadrille
    cadrille_eval_dir = results_dir / "cadrille_eval" / dataset
    cadrille_eval_dir.mkdir(exist_ok=True, parents=True)
    
    success = run_command([
        "python", "evaluate.py",
        "--gt-mesh-path", str(Path(data_path) / dataset),
        "--pred-py-path", str(cadrille_output),
        "--pred-eval-path", str(cadrille_eval_dir),
        "--n-points", "8192"
    ], "Оценка Cadrille")
    
    if not success:
        return
    
    # 3. Сравнение результатов
    comparison_dir = results_dir / "comparison" / dataset
    comparison_dir.mkdir(exist_ok=True, parents=True)
    
    success = run_command([
        "python", "compare_results.py",
        "--cad-recode-results", str(cad_recode_eval_dir / "summary.txt"),
        "--cadrille-results", str(cadrille_eval_dir / "summary.txt"),
        "--output-dir", str(comparison_dir),
        "--dataset", dataset
    ], "Сравнение результатов")
    
    if success:
        print_header("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        comparison_json_path = comparison_dir / "comparison_results.json"
        if comparison_json_path.exists():
            with open(comparison_json_path, 'r') as f:
                results = json.load(f)
                logger.info(f"Модели: {', '.join(results['models'])}")
                metrics = results.get('metrics', {})
                if 'chamfer_distances_mm' in metrics:
                    logger.info(f"Chamfer Distance (мм): {', '.join([f'{cd:.4f}' for cd in metrics['chamfer_distances_mm']])}")
                if 'iou_values' in metrics:
                    logger.info(f"IoU: {', '.join([f'{iou:.4f}' for iou in metrics['iou_values']])}")
                logger.info(f"\nПодробные результаты сохранены в {comparison_dir}")
        else:
            logger.warning(f"⚠️ Файл результатов не найден: {comparison_json_path}")

def main():
    """
    Главная функция для запуска конвейера из командной строки.
    
    Парсит аргументы командной строки и запускает полный конвейер тестирования
    и оценки моделей CAD. Выполняет последовательность этапов:
    1. Тестирование модели Cadrille (если указано)
    2. Тестирование модели CAD-Recode (если указано)
    3. Оценка результатов
    4. Сравнение результатов (если указано)
    
    Args:
        args (Namespace): Аргументы командной строки, содержащие:
            - data_path: путь к данным
            - split: название датасета
            - cadrille_checkpoint: путь к модели Cadrille
            - cad_recode_model_path: путь к модели CAD-Recode
            - output_dir: директория для результатов
            - device: устройство (cuda/cpu)
            - max_samples: максимальное количество образцов
            - compare: флаг для сравнения результатов
            - И другие параметры
    
    Note:
        Функция использует argparse для парсинга аргументов и передает их
        в run_pipeline() для выполнения конвейера. Обрабатывает ошибки и
        выводит информацию о выполнении.
    
    Example:
        >>> # Запуск из командной строки:
        >>> # python pipeline.py --data-path /workspace/data --split deepcad_test_mesh ...
    """
    parser = ArgumentParser(description='Конвейер тестирования и оценки моделей CAD-Recode и Cadrille')
    parser.add_argument('--data-path', type=str, default='/workspace/data',
                        help='Путь к директории с данными')
    parser.add_argument('--models-dir', type=str, default='/workspace/models',
                        help='Путь к директории с моделями')
    parser.add_argument('--results-dir', type=str, default='/workspace/results',
                        help='Путь для сохранения результатов')
    parser.add_argument('--dataset', type=str, default='deepcad_test_mesh',
                        choices=['deepcad_test_mesh', 'fusion360_test_mesh'],
                        help='Датасет для тестирования')
    parser.add_argument('--n-samples', type=int, default=1,
                        help='Количество генераций на объект')
    parser.add_argument('--max-samples', type=int, default=50,
                        help='Максимальное количество объектов для обработки')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство для вычислений (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Проверка доступности GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA недоступна, переключаемся на CPU")
        args.device = 'cpu'
    
    if torch.cuda.is_available():
        logger.info(f"GPU доступен: {torch.cuda.get_device_name(0)}")
        logger.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Запуск конвейера
    run_pipeline(
        args.data_path,
        args.models_dir,
        args.results_dir,
        args.dataset,
        args.n_samples,
        args.max_samples,
        args.device
    )

if __name__ == "__main__":
    main()
    