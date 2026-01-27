#!/usr/bin/env python3
"""
Сравнение результатов оценки моделей CAD-Recode и Cadrille
с визуализацией метрик
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_summary_file(file_path: str) -> dict:
    """
    Парсит файл summary.txt и извлекает метрики оценки моделей.
    
    Читает текстовый файл с результатами оценки модели и извлекает числовые
    значения метрик (Chamfer Distance, IoU, Invalidity Ratio и другие).
    Поддерживает различные форматы записи метрик (с единицами измерения,
    процентами, обычными числами).
    
    Алгоритм парсинга:
        1. Чтение файла построчно
        2. Поиск строк формата "ключ: значение"
        3. Извлечение числового значения с учетом единиц измерения
        4. Преобразование в float и сохранение в словаре
    
    Args:
        file_path (str): Путь к файлу summary.txt с результатами оценки.
            Должен существовать и быть читаемым. Формат: "Метрика: значение".
    
    Returns:
        dict: Словарь с метриками, где ключи - названия метрик (str),
            значения - числовые значения метрик (float).
            Возвращает пустой словарь, если файл не найден или не содержит метрик.
    
    Raises:
        IOError: Если не удалось прочитать файл (обрабатывается внутри функции)
    
    Note:
        Функция обрабатывает различные форматы значений:
        - С единицами измерения: "Chamfer Distance: 0.1234 мм" -> 0.1234
        - С процентами: "Invalidity Ratio: 5.5%" -> 5.5
        - Обычные числа: "IoU: 0.8567" -> 0.8567
        Строки без разделителя ':' или с невалидными значениями пропускаются.
    
    Example:
        >>> metrics = parse_summary_file('/workspace/results/summary.txt')
        >>> print(metrics)
        {'Средний Chamfer Distance': 0.1234, 'Средний IoU': 0.8567, 'Invalidity Ratio': 5.5}
    """
    metrics = {}
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ Файл не найден: {file_path}")
        return metrics
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Преобразование значений в числа
            try:
                if 'mm' in value or 'мм' in value:
                    # Chamfer Distance в миллиметрах
                    num_value = float(value.split()[0])
                    metrics[key] = num_value
                elif '%' in value:
                    # Процентные значения
                    num_value = float(value.replace('%', ''))
                    metrics[key] = num_value
                else:
                    # Обычные числовые значения
                    num_value = float(value)
                    metrics[key] = num_value
            except (ValueError, IndexError):
                continue
    
    return metrics

def compare_and_visualize(cad_recode_results: str, cadrille_results: str, 
                          output_dir: str, dataset: str) -> None:
    """
    Сравнивает результаты оценки моделей CAD-Recode и Cadrille с визуализацией.
    
    Загружает метрики обеих моделей, сравнивает их и создает визуализации:
    столбчатые диаграммы для отдельных метрик и радарную диаграмму для
    комплексного сравнения. Сохраняет результаты в различных форматах.
    
    Процесс выполнения:
        1. Загрузка метрик из summary.txt файлов обеих моделей
        2. Извлечение ключевых метрик (Chamfer Distance, IoU, Invalidity Ratio)
        3. Создание столбчатых диаграмм для каждой метрики
        4. Создание радарной диаграммы для комплексного сравнения
        5. Сохранение результатов в JSON и текстовый отчет
    
    Args:
        cad_recode_results (str): Путь к файлу summary.txt с результатами CAD-Recode.
            Должен содержать метрики оценки модели CAD-Recode.
        cadrille_results (str): Путь к файлу summary.txt с результатами Cadrille.
            Должен содержать метрики оценки модели Cadrille.
        output_dir (str): Директория для сохранения результатов сравнения.
            Будет создана автоматически, если не существует. В ней будут сохранены:
            - chamfer_distance_comparison.png: сравнение Chamfer Distance
            - iou_comparison.png: сравнение IoU
            - invalidity_ratio_comparison.png: сравнение Invalidity Ratio
            - radar_comparison.png: радарная диаграмма
            - comparison_results.json: данные сравнения в JSON формате
            - comparison_report.txt: текстовый отчет с выводами
        dataset (str): Название датасета для заголовков графиков и отчетов.
            Используется только для информационных целей.
    
    Returns:
        None: Функция не возвращает значение, но создает файлы с результатами
            в указанной директории output_dir.
    
    Raises:
        FileNotFoundError: Если не найдены файлы summary.txt
        ValueError: Если не удалось загрузить метрики из файлов
    
    Note:
        Функция создает 4 визуализации:
        - Столбчатые диаграммы для Chamfer Distance, IoU и Invalidity Ratio
        - Радарная диаграмма для комплексного сравнения всех метрик
        Метрики нормализуются для радарной диаграммы (Chamfer Distance и
        Invalidity Ratio инвертируются, так как "меньше лучше").
        Текстовый отчет содержит автоматические выводы о том, какая модель
        показывает лучшие результаты по каждой метрике.
    
    Example:
        >>> compare_and_visualize(
        ...     cad_recode_results='/workspace/results/cad_recode_eval/summary.txt',
        ...     cadrille_results='/workspace/results/cadrille_eval/summary.txt',
        ...     output_dir='/workspace/results/comparison',
        ...     dataset='deepcad_test_mesh'
        ... )
    """
    # Создание директории для вывода
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Загрузка результатов
    cad_recode_metrics = parse_summary_file(cad_recode_results)
    cadrille_metrics = parse_summary_file(cadrille_results)
    
    if not cad_recode_metrics or not cadrille_metrics:
        logger.error("❌ Не удалось загрузить результаты для сравнения")
        return
    
    logger.info("📊 Загруженные метрики:")
    logger.info(f"CAD-Recode: {cad_recode_metrics}")
    logger.info(f"Cadrille: {cadrille_metrics}")
    
    # Подготовка данных для визуализации
    models = ['CAD-Recode', 'Cadrille']
    chamfer_distances = []
    iou_values = []
    invalidity_ratios = []
    
    # Chamfer Distance (в миллиметрах)
    for metrics in [cad_recode_metrics, cadrille_metrics]:
        if 'Средний Chamfer Distance' in metrics:
            chamfer_distances.append(metrics['Средний Chamfer Distance'])
        elif 'Mean Chamfer Distance' in metrics:
            chamfer_distances.append(metrics['Mean Chamfer Distance'])
        else:
            chamfer_distances.append(0)
    
    # IoU
    for metrics in [cad_recode_metrics, cadrille_metrics]:
        if 'Средний IoU' in metrics:
            iou_values.append(metrics['Средний IoU'])
        elif 'Mean IoU' in metrics:
            iou_values.append(metrics['Mean IoU'])
        else:
            iou_values.append(0)
    
    # Invalidity Ratio
    for metrics in [cad_recode_metrics, cadrille_metrics]:
        if 'Invalidity Ratio' in metrics:
            invalidity_ratios.append(metrics['Invalidity Ratio'])
        else:
            invalidity_ratios.append(0)
    
    # Визуализация Chamfer Distance
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, chamfer_distances, color=['#3498db', '#2ecc71'])
    plt.title(f'Сравнение Chamfer Distance на {dataset}\n(меньше лучше)', fontsize=14, fontweight='bold')
    plt.ylabel('Chamfer Distance (мм)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Добавление значений над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    chamfer_path = output_dir / 'chamfer_distance_comparison.png'
    plt.savefig(chamfer_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Визуализация IoU
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, iou_values, color=['#3498db', '#2ecc71'])
    plt.title(f'Сравнение IoU на {dataset}\n(больше лучше)', fontsize=14, fontweight='bold')
    plt.ylabel('IoU', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Добавление значений над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    iou_path = output_dir / 'iou_comparison.png'
    plt.savefig(iou_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Визуализация Invalidity Ratio
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, invalidity_ratios, color=['#3498db', '#2ecc71'])
    plt.title(f'Сравнение Invalidity Ratio на {dataset}\n(меньше лучше)', fontsize=14, fontweight='bold')
    plt.ylabel('Invalidity Ratio (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Добавление значений над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    invalidity_path = output_dir / 'invalidity_ratio_comparison.png'
    plt.savefig(invalidity_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Радарная диаграмма для комплексного сравнения
    plt.figure(figsize=(10, 8))
    
    # Нормализация метрик для радарной диаграммы
    # Все метрики приводятся к диапазону [0, 1] для визуализации на одной шкале
    
    # Chamfer Distance: инвертируем, так как меньше лучше
    # Нормализуем так, чтобы лучшее значение (минимальное) было ближе к 1
    if chamfer_distances and max(chamfer_distances) > 0:
        max_cd = max(chamfer_distances) * 1.1  # Добавляем 10% запаса для визуализации
        min_cd = 0
        # Инвертируем: (max - value) / (max - min), чтобы меньшее значение давало больший нормализованный результат
        cd_normalized = [(max_cd - cd) / (max_cd - min_cd) if (max_cd - min_cd) > 0 else 0.5 for cd in chamfer_distances]
    else:
        cd_normalized = [0.5, 0.5]  # Значения по умолчанию при отсутствии данных
    
    # IoU: нормализуем напрямую, так как больше лучше
    # IoU уже находится в диапазоне [0, 1], просто нормализуем к [0, 1]
    max_iou = 1.0
    min_iou = 0
    iou_normalized = [iou / (max_iou - min_iou) if (max_iou - min_iou) > 0 else iou for iou in iou_values]
    
    # Invalidity Ratio: инвертируем, так как меньше лучше
    # Нормализуем так, чтобы лучшее значение (минимальное) было ближе к 1
    if invalidity_ratios and max(invalidity_ratios) > 0:
        max_invalid = max(invalidity_ratios) * 1.1  # Добавляем 10% запаса
        min_invalid = 0
        # Инвертируем для согласованности с другими метриками "меньше лучше"
        invalid_normalized = [(max_invalid - invalid) / (max_invalid - min_invalid) if (max_invalid - min_invalid) > 0 else 0.5 for invalid in invalidity_ratios]
    else:
        invalid_normalized = [0.5, 0.5]  # Значения по умолчанию
    
    # Данные для радарной диаграммы
    metrics_names = ['Chamfer\nDistance', 'IoU', 'Invalidity\nRatio']
    model1_values = [cd_normalized[0], iou_normalized[0], invalid_normalized[0]]
    model2_values = [cd_normalized[1], iou_normalized[1], invalid_normalized[1]]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    model1_values += model1_values[:1]
    model2_values += model2_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, model1_values, 'o-', linewidth=2, label='CAD-Recode', color='#3498db')
    ax.plot(angles, model2_values, 'o-', linewidth=2, label='Cadrille', color='#2ecc71')
    ax.fill(angles, model1_values, alpha=0.1, color='#3498db')
    ax.fill(angles, model2_values, alpha=0.1, color='#2ecc71')
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], metrics_names, size=12)
    ax.tick_params(axis='x', pad=20)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.0)
    
    plt.title(f'Комплексное сравнение моделей на {dataset}', size=14, weight='bold', position=(0.5, 1.1))
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    radar_path = output_dir / 'radar_comparison.png'
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохранение сравнительных данных в JSON
    comparison_data = {
        'models': models,
        'dataset': dataset,
        'metrics': {
            'chamfer_distances_mm': [float(cd) for cd in chamfer_distances],
            'iou_values': [float(iou) for iou in iou_values],
            'invalidity_ratios_percent': [float(ir) for ir in invalidity_ratios]
        },
        'plots': {
            'chamfer_distance': str(chamfer_path),
            'iou': str(iou_path),
            'invalidity_ratio': str(invalidity_path),
            'radar': str(radar_path)
        }
    }
    
    json_path = output_dir / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Создание текстового отчета
    report_path = output_dir / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"ОТЧЕТ О СРАВНЕНИИ МОДЕЛЕЙ НА ДАТАСЕТЕ: {dataset}\n")
        f.write("="*60 + "\n\n")
        
        f.write("МЕТРИКИ CAD-RECODE:\n")
        for key, value in cad_recode_metrics.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nМЕТРИКИ CADRILLE:\n")
        for key, value in cadrille_metrics.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("ВЫВОДЫ:\n")
        f.write("="*60 + "\n")
        
        # Сравнение Chamfer Distance
        if chamfer_distances[0] < chamfer_distances[1]:
            f.write(f"✅ CAD-Recode показывает лучший результат по Chamfer Distance\n")
            f.write(f"   Разница: {chamfer_distances[1] - chamfer_distances[0]:.4f} мм\n")
        elif chamfer_distances[0] > chamfer_distances[1]:
            f.write(f"✅ Cadrille показывает лучший результат по Chamfer Distance\n")
            f.write(f"   Разница: {chamfer_distances[0] - chamfer_distances[1]:.4f} мм\n")
        else:
            f.write("📊 Модели показывают одинаковые результаты по Chamfer Distance\n")
        
        # Сравнение IoU
        if iou_values[0] > iou_values[1]:
            f.write(f"✅ CAD-Recode показывает лучший результат по IoU\n")
            f.write(f"   Разница: {iou_values[0] - iou_values[1]:.4f}\n")
        elif iou_values[0] < iou_values[1]:
            f.write(f"✅ Cadrille показывает лучший результат по IoU\n")
            f.write(f"   Разница: {iou_values[1] - iou_values[0]:.4f}\n")
        else:
            f.write("📊 Модели показывают одинаковые результаты по IoU\n")
        
        # Сравнение Invalidity Ratio
        if invalidity_ratios[0] < invalidity_ratios[1]:
            f.write(f"✅ CAD-Recode показывает лучший результат по Invalidity Ratio\n")
            f.write(f"   Разница: {invalidity_ratios[1] - invalidity_ratios[0]:.2f}%\n")
        elif invalidity_ratios[0] > invalidity_ratios[1]:
            f.write(f"✅ Cadrille показывает лучший результат по Invalidity Ratio\n")
            f.write(f"   Разница: {invalidity_ratios[0] - invalidity_ratios[1]:.2f}%\n")
        else:
            f.write("📊 Модели показывают одинаковые результаты по Invalidity Ratio\n")
    
    logger.info("\n" + "="*60)
    logger.info("✅ СРАВНЕНИЕ ЗАВЕРШЕНО")
    logger.info("="*60)
    logger.info(f"Графики сохранены в: {output_dir}")
    logger.info(f"Chamfer Distance: {chamfer_path}")
    logger.info(f"IoU: {iou_path}")
    logger.info(f"Invalidity Ratio: {invalidity_path}")
    logger.info(f"Радарная диаграмма: {radar_path}")
    logger.info(f"JSON данные: {json_path}")
    logger.info(f"Текстовый отчет: {report_path}")
    logger.info("="*60)

if __name__ == "__main__":
    parser = ArgumentParser(description='Сравнение результатов оценки моделей')
    parser.add_argument('--cad-recode-results', type=str, required=True,
                        help='Путь к summary.txt для CAD-Recode')
    parser.add_argument('--cadrille-results', type=str, required=True,
                        help='Путь к summary.txt для Cadrille')
    parser.add_argument('--output-dir', type=str, default='/workspace/results/comparison',
                        help='Директория для сохранения результатов сравнения')
    parser.add_argument('--dataset', type=str, default='deepcad_test_mesh',
                        help='Название датасета для заголовков')
    
    args = parser.parse_args()
    
    compare_and_visualize(
        args.cad_recode_results,
        args.cadrille_results,
        args.output_dir,
        args.dataset
    )
    