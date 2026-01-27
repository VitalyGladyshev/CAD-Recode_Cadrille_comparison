#!/usr/bin/env python3
"""
Тестирование модели CAD-Recode с генерацией CadQuery кодов
Совместим с форматом выходных данных Cadrille для объективного сравнения
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import trimesh
import open3d
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Импорты для модели
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Model, PreTrainedModel, Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from pytorch3d.ops import sample_farthest_points
import torch.nn as nn


class FourierPointEncoder(nn.Module):
    """
    Кодировщик точечных облаков для CAD-Recode с использованием Fourier features.
    
    Преобразует координаты точек из точечного облака в embeddings для модели.
    Использует Fourier positional encoding для создания высокочастотных признаков,
    что позволяет модели лучше понимать пространственные отношения в геометрии.
    
    Архитектура:
        1. Создание Fourier features: умножение координат на частоты и применение sin/cos
        2. Объединение исходных координат с Fourier features
        3. Линейная проекция в пространство скрытых состояний модели
    
    Attributes:
        frequencies (torch.Tensor): Буфер с частотами для Fourier преобразования.
            Форма: (8,). Частоты: 2^0, 2^1, ..., 2^7.
        projection (nn.Linear): Линейный слой для проекции в hidden_size измерений.
            Входная размерность: 51 (3 исходных + 8*2*3 Fourier features).
    
    Note:
        Использует register_buffer для сохранения frequencies как части состояния
        модели без необходимости обновления через оптимизатор.
        Выходная размерность: hidden_size (обычно 1536 для Qwen2 моделей).
    """
    
    def __init__(self, hidden_size):
        """
        Инициализирует FourierPointEncoder.
        
        Args:
            hidden_size (int): Размерность скрытого пространства модели.
                Должен совпадать с hidden_size конфигурации модели.
        """
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)
    
    def forward(self, points):
        """
        Кодирует точечное облако в embeddings.
        
        Применяет Fourier преобразование к координатам точек и проектирует
        результат в пространство скрытых состояний модели.
        
        Args:
            points (torch.Tensor): Точечное облако формы (batch, n_points, 3).
                Содержит координаты (x, y, z) для каждой точки.
        
        Returns:
            torch.Tensor: Embeddings формы (batch, n_points, hidden_size).
                Каждая точка представлена вектором размерности hidden_size.
        """
        x = points
        # Создаем Fourier features: умножаем координаты на частоты
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        # Объединяем исходные координаты с sin/cos преобразованиями
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        # Проектируем в пространство скрытых состояний
        x = self.projection(x)
        return x


class CADRecode(Qwen2ForCausalLM):
    """
    Модель CAD-Recode для генерации CadQuery кодов из точечных облаков.
    
    Расширяет Qwen2ForCausalLM для поддержки точечных облаков через
    FourierPointEncoder. Модель принимает точечное облако 3D объекта и
    генерирует соответствующий CadQuery код, описывающий его геометрию.
    
    Архитектура:
        - Базовый трансформер: Qwen2Model (языковая модель)
        - Point encoder: FourierPointEncoder для кодирования точечных облаков
        - Language model head: генерация токенов CadQuery кода
    
    Процесс работы:
        1. Точечное облако кодируется через FourierPointEncoder
        2. Point embeddings встраиваются в позиции pad токенов входной последовательности
        3. Трансформер обрабатывает объединенную последовательность
        4. Language model head генерирует токены CadQuery кода
    
    Attributes:
        model (Qwen2Model): Базовый трансформер для обработки последовательностей
        vocab_size (int): Размер словаря токенизатора
        lm_head (nn.Linear): Головка для генерации токенов
        point_encoder (FourierPointEncoder): Энкодер для точечных облаков
    
    Note:
        Point encoder создается в float32 для обеспечения численной стабильности
        при вычислении Fourier features, затем dtype возвращается к bfloat16.
        Модель наследует функциональность генерации от Qwen2ForCausalLM.
    
    Example:
        >>> from test_cad_recode import CADRecode
        >>> from transformers import Qwen2Config
        >>> config = Qwen2Config.from_pretrained('model_path')
        >>> model = CADRecode(config)
        >>> # Генерация кода из точечного облака
        >>> outputs = model.generate(
        ...     input_ids=input_ids,
        ...     attention_mask=attention_mask,
        ...     point_cloud=point_cloud,
        ...     max_new_tokens=768
        ... )
    """
    config_class = Qwen2Config

    def __init__(self, config):
        """
        Инициализирует модель CAD-Recode.
        
        Args:
            config (Qwen2Config): Конфигурация модели.
                Должна содержать параметры hidden_size, vocab_size и другие.
        
        Note:
            Важно вызывать PreTrainedModel.__init__ вместо super().__init__
            для корректной инициализации базового класса.
            Point encoder создается в float32 для стабильности вычислений.
        """
        # Важно: вызываем именно PreTrainedModel.__init__, как в оригинале
        PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Создаём point_encoder в float32 для численной стабильности Fourier features
        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                point_cloud=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                cache_position=None,
                **kwargs):
        """
        Прямой проход модели CAD-Recode.
        
        Обрабатывает точечные облака и текстовые входные данные, генерирует
        логиты для предсказания следующего токена. Поддерживает кэширование
        для ускорения генерации.
        
        Алгоритм обработки (только на первом проходе):
            1. Создание текстовых embeddings из input_ids
            2. Кодирование точечного облака через FourierPointEncoder
            3. Встраивание point embeddings в позиции pad токенов (attention_mask == -1)
            4. Обновление attention_mask (замена -1 на 1)
            5. Передача inputs_embeds в базовую модель (input_ids = None)
        
        Args:
            input_ids (torch.Tensor, optional): Токенизированные входные данные.
                Форма: (batch_size, seq_length). Содержит pad токены для point cloud.
                По умолчанию None.
            attention_mask (torch.Tensor, optional): Маска внимания.
                Форма: (batch_size, seq_length). Значение -1 указывает позиции point cloud.
                По умолчанию None.
            point_cloud (torch.Tensor, optional): Точечное облако для кодирования.
                Форма: (batch_size, n_points, 3). По умолчанию None.
            position_ids (torch.Tensor, optional): Позиционные индексы.
                Вычисляются автоматически, если не заданы. По умолчанию None.
            past_key_values (Cache, optional): Кэш для ускорения генерации.
                По умолчанию None.
            inputs_embeds (torch.Tensor, optional): Предвычисленные embeddings.
                По умолчанию None.
            labels (torch.Tensor, optional): Labels для вычисления loss.
                По умолчанию None.
            use_cache (bool, optional): Использовать ли кэширование.
                По умолчанию None.
            output_attentions (bool, optional): Возвращать ли attention weights.
                По умолчанию None.
            output_hidden_states (bool, optional): Возвращать ли скрытые состояния.
                По умолчанию None.
            return_dict (bool, optional): Возвращать ли объект вместо tuple.
                По умолчанию None.
            cache_position (torch.Tensor, optional): Позиции в кэше.
                По умолчанию None.
            **kwargs: Дополнительные аргументы для базовой модели.
        
        Returns:
            CausalLMOutputWithPast или tuple: Если return_dict=True, возвращает
                объект с полями:
                - loss (torch.Tensor, optional): Cross-entropy loss (если labels заданы)
                - logits (torch.Tensor): Логиты для предсказания токенов
                - past_key_values (Cache, optional): Кэш для следующей итерации
                - hidden_states (tuple, optional): Скрытые состояния всех слоев
                - attentions (tuple, optional): Attention weights всех слоев
        
        Raises:
            AssertionError: Если inputs_embeds заданы на первом проходе
            ValueError: Если параметры имеют недопустимые значения
        
        Note:
            Point cloud embeddings встраиваются только на первом проходе (когда
            past_key_values пуст). На последующих итерациях генерации используются
            только текстовые токены из кэша.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Объединение point и text embeddings - ТОЛЬКО на первом проходе генерации
        # На последующих итерациях используются только текстовые токены из кэша
        if past_key_values is None or (hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0):
            assert inputs_embeds is None
            
            # Шаг 1: Создаём embeddings для ВСЕХ input_ids (включая pad токены для point cloud)
            # Pad токены в начале последовательности будут заменены на point embeddings
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            # Шаг 2: Кодируем point cloud через FourierPointEncoder
            # Конвертируем в dtype модели для совместимости
            point_embeds = self.point_encoder(point_cloud).to(inputs_embeds.dtype)
            
            # Шаг 3: Заменяем embeddings в позициях point cloud (где attention_mask == -1)
            # attention_mask == -1 указывает на позиции pad токенов, которые нужно заменить
            # Reshape нужен для правильного выравнивания размерностей при индексации
            inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
            
            # Шаг 4: Обновляем attention_mask (заменяем -1 на 1 для point cloud позиций)
            # Теперь point embeddings будут участвовать в вычислении внимания
            attention_mask = attention_mask.clone()
            attention_mask[attention_mask == -1] = 1
            
            # Шаг 5: Обнуляем input_ids и position_ids для использования inputs_embeds
            # Базовая модель будет использовать inputs_embeds вместо input_ids
            # Это необходимо, так как мы модифицировали embeddings напрямую
            input_ids = None
            position_ids = None

        # Вызов базовой модели
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Подготавливает входные данные для генерации.
        
        Расширяет базовый метод prepare_inputs_for_generation для передачи
        параметра point_cloud через все итерации генерации. Это критически важно
        для корректной работы модели, так как point_cloud должен быть доступен
        на каждой итерации генерации.
        
        Args:
            *args: Позиционные аргументы базового метода
            **kwargs: Именованные аргументы, могут содержать:
                - point_cloud (torch.Tensor): Точечное облако для генерации
        
        Returns:
            dict: Подготовленные входные данные с добавленным полем point_cloud
                (если было передано в kwargs)
        
        Note:
            Метод вызывается автоматически на каждой итерации генерации.
            Необходимо для сохранения point_cloud в model_inputs для передачи
            в метод forward на каждой итерации.
        """
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs['point_cloud'] = kwargs.get('point_cloud')
        return model_inputs


def mesh_to_point_cloud(mesh, n_points=256, n_pre_points=8192):
    """
    Конвертирует 3D mesh в точечное облако с использованием farthest point sampling.
    
    Использует двухэтапный процесс для создания равномерно распределенного
    точечного облака: сначала выполняется предварительное сэмплирование большого
    количества точек с поверхности mesh, затем farthest point sampling выбирает
    наиболее равномерно распределенные точки.
    
    Args:
        mesh (trimesh.Trimesh): Исходный 3D mesh для конвертации.
            Должен быть валидной моделью с вершинами и гранями.
        n_points (int, optional): Количество точек в итоговом точечном облаке.
            Типичные значения: 256, 512, 1024. По умолчанию 256.
        n_pre_points (int, optional): Количество точек для предварительного
            сэмплирования перед farthest point sampling. Больше точек дают
            лучший результат, но требуют больше вычислений.
            По умолчанию 8192.
    
    Returns:
        np.ndarray: Массив точек формы (n_points, 3) с координатами (x, y, z).
            Точки представлены в той же системе координат, что и исходный mesh.
    
    Raises:
        ValueError: Если mesh пустой или невалидный.
        RuntimeError: Если не удалось выполнить сэмплирование поверхности.
    
    Note:
        Farthest point sampling обеспечивает более равномерное распределение точек
        по сравнению с простым случайным сэмплированием, что важно для качества
        представления геометрии модели.
    
    Example:
        >>> import trimesh
        >>> mesh = trimesh.load('model.stl')
        >>> point_cloud = mesh_to_point_cloud(mesh, n_points=256)
        >>> print(f"Точечное облако: {point_cloud.shape}")
        Точечное облако: (256, 3)
    """
    # Предварительное сэмплирование точек с поверхности mesh
    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    # Farthest point sampling для выбора равномерно распределенных точек
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    return np.asarray(vertices[ids])


def extract_code(tokenizer, batch_ids):
    """
    Извлекает CadQuery код из токенизированного вывода модели.
    
    Удаляет все служебные токены (<|im_start|>, <|endoftext|>, <|im_end|>)
    и извлекает чистый Python код с CadQuery импортами и командами.
    Обрабатывает различные форматы вывода модели для обеспечения корректного
    извлечения кода.
    
    Алгоритм:
        1. Декодирование токенов в строку (без пропуска специальных токенов)
        2. Поиск и удаление <|im_start|> маркера
        3. Обрезка до <|endoftext|> маркера
        4. Удаление всех оставшихся служебных токенов
        5. Поиск начала кода (import cadquery или первый import)
    
    Args:
        tokenizer: Токенизатор модели (AutoTokenizer).
            Должен поддерживать batch_decode метод.
        batch_ids (torch.Tensor): Токенизированный вывод модели.
            Форма: (batch_size, seq_length) или (seq_length,).
            Содержит token IDs, включая служебные токены.
    
    Returns:
        str: Извлеченный CadQuery код без служебных токенов.
            Начинается с 'import cadquery' или другого import statement.
            Пустая строка, если код не найден.
    
    Note:
        Функция обрабатывает различные форматы вывода модели и пытается
        найти начало кода даже если структура вывода нестандартная.
        Все служебные токены удаляются для получения чистого Python кода.
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('Qwen2-1.5B')
        >>> batch_ids = model.generate(...)
        >>> code = extract_code(tokenizer, batch_ids)
        >>> print(code[:50])
        import cadquery as cq
        w0=cq.Workplane('XY')...
    """
    # Декодируем с пропуском специальных токенов
    py_string = tokenizer.batch_decode(batch_ids, skip_special_tokens=False)[0]
    
    # Удаляем все служебные токены
    # Сначала находим и удаляем <|im_start|>
    if '<|im_start|>' in py_string:
        # Находим начало после <|im_start|>
        start_idx = py_string.find('<|im_start|>')
        if start_idx != -1:
            py_string = py_string[start_idx + len('<|im_start|>'):]
    
    # Удаляем <|endoftext|> и все после него
    if '<|endoftext|>' in py_string:
        end_idx = py_string.find('<|endoftext|>')
        if end_idx != -1:
            py_string = py_string[:end_idx]
    
    # Удаляем другие возможные служебные токены
    py_string = py_string.replace('<|im_end|>', '')
    py_string = py_string.replace('<|endoftext|>', '')
    py_string = py_string.replace('<|im_start|>', '')
    
    # Если не нашли import cadquery, пробуем найти его
    if 'import cadquery' not in py_string:
        # Ищем любую строку, начинающуюся с import
        import_idx = py_string.find('import')
        if import_idx != -1:
            py_string = py_string[import_idx:]
    
    return py_string.strip()


class STLDataset(Dataset):
    """
    Датасет для загрузки STL файлов для тестирования модели CAD-Recode.
    
    Простой датасет, который сканирует директорию с STL файлами и предоставляет
    доступ к ним по индексу. Используется для инференса модели на тестовых данных.
    
    Attributes:
        data_path (Path): Путь к корневой директории с данными
        split (str): Название поддиректории с STL файлами
        stl_files (list): Список путей к STL файлам
    
    Example:
        >>> dataset = STLDataset('/workspace/data', 'deepcad_test_mesh', max_samples=50)
        >>> item = dataset[0]
        >>> print(item)
        {'file_path': '/workspace/data/deepcad_test_mesh/file.stl', 'file_name': 'file'}
    """
    
    def __init__(self, data_path, split, max_samples=None):
        """
        Инициализирует датасет STLDataset.
        
        Args:
            data_path (str or Path): Путь к корневой директории с данными
            split (str): Название поддиректории с STL файлами для тестирования
            max_samples (int, optional): Максимальное количество файлов для загрузки.
                Если None, загружаются все найденные файлы. По умолчанию None.
        
        Raises:
            FileNotFoundError: Если директория data_path/split не существует
        """
        self.data_path = Path(data_path)
        self.split = split
        self.stl_files = list((self.data_path / split).glob("*.stl"))
        
        if max_samples and max_samples < len(self.stl_files):
            self.stl_files = self.stl_files[:max_samples]
    
    def __len__(self):
        """
        Возвращает количество STL файлов в датасете.
        
        Returns:
            int: Количество файлов в датасете
        """
        return len(self.stl_files)

    def __getitem__(self, idx):
        """
        Получает элемент датасета по индексу.
        
        Args:
            idx (int): Индекс элемента в датасете
        
        Returns:
            dict: Словарь с ключами:
                - 'file_path' (str): Полный путь к STL файлу
                - 'file_name' (str): Имя файла без расширения
        
        Raises:
            IndexError: Если индекс выходит за границы датасета
        """
        stl_file = self.stl_files[idx]
        return {
            'file_path': str(stl_file),
            'file_name': stl_file.stem
        }


def run_inference_batch(model, tokenizer, batch, output_dir, device, sample_idx=0):
    """
    Генерирует CadQuery коды для батча данных.
    
    Обрабатывает каждый элемент батча: загружает STL файл, конвертирует в точечное
    облако, генерирует CadQuery код с помощью модели и сохраняет результат в файл.
    Поддерживает обработку как отдельных элементов, так и батчей.
    
    Алгоритм для каждого элемента:
        1. Загрузка и нормализация mesh из STL файла
        2. Конвертация mesh в точечное облако (256 точек)
        3. Подготовка входных данных (pad токены + <|im_start|> токен)
        4. Генерация CadQuery кода моделью
        5. Извлечение и очистка кода от служебных токенов
        6. Сохранение кода в файл
    
    Args:
        model (CADRecode): Модель CAD-Recode для генерации кодов.
            Должна быть загружена и переведена в режим eval.
        tokenizer (AutoTokenizer): Токенизатор модели.
            Должен соответствовать модели и поддерживать encode/decode методы.
        batch (dict or list): Батч данных от DataLoader. Может быть:
            - dict: словарь с батчированными полями ('file_path', 'file_name')
            - list: список словарей с элементами датасета
        output_dir (Path): Директория для сохранения сгенерированных .py файлов.
            Должна существовать или быть создана заранее.
        device (str): Устройство для вычислений ('cuda' или 'cpu').
            Должно соответствовать устройству, на котором загружена модель.
        sample_idx (int, optional): Индекс сэмпла для множественных генераций.
            Используется в имени файла: '{file_name}+{sample_idx}.py'.
            По умолчанию 0.
    
    Returns:
        list: Список словарей с результатами генерации для каждого элемента батча.
            Каждый словарь содержит:
            - 'file_name' (str): Имя файла без расширения
            - 'code' (str): Сгенерированный CadQuery код
    
    Raises:
        FileNotFoundError: Если STL файл не найден
        RuntimeError: Если не удалось загрузить mesh или выполнить генерацию
        ValueError: Если mesh невалидный или пустой
    
    Note:
        Mesh нормализуется перед конвертацией в точечное облако: центрируется
        относительно начала координат и масштабируется до размера [-1, 1].
        Это обеспечивает единообразную обработку моделей разного размера.
    
    Example:
        >>> from pathlib import Path
        >>> output_dir = Path('/workspace/results')
        >>> results = run_inference_batch(
        ...     model, tokenizer, batch, output_dir, device='cuda', sample_idx=0
        ... )
        >>> print(f"Обработано {len(results)} файлов")
        Обработано 8 файлов
    """
    results = []
    
    # DataLoader возвращает словарь с батчированными полями
    if isinstance(batch, dict):
        file_paths = batch['file_path'] if isinstance(batch['file_path'], list) else [batch['file_path']]
        file_names = batch['file_name'] if isinstance(batch['file_name'], list) else [batch['file_name']]
    else:
        file_paths = [item['file_path'] for item in batch]
        file_names = [item['file_name'] for item in batch]
    
    # Обрабатываем каждый элемент батча
    for file_path, file_name in zip(file_paths, file_names):
        # Загрузка и обработка STL файла
        gt_mesh = trimesh.load_mesh(file_path)
        
        # Нормализация mesh
        gt_mesh.apply_translation(-(gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2.0)
        gt_mesh.apply_scale(2.0 / max(gt_mesh.extents))
        
        # Генерация точечного облака
        point_cloud = mesh_to_point_cloud(gt_mesh, n_points=256)
        
        # Подготовка входных данных
        input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
        attention_mask = [-1] * len(point_cloud) + [1]
        
        # Генерация кода
        with torch.no_grad():
            batch_ids = model.generate(
                input_ids=torch.tensor(input_ids).unsqueeze(0).to(device),
                attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(device),
                point_cloud=torch.tensor(point_cloud.astype(np.float32)).unsqueeze(0).to(device),
                max_new_tokens=768,
                pad_token_id=tokenizer.pad_token_id)
        
        # Декодирование результата - используем исправленную функцию
        py_string = extract_code(tokenizer, batch_ids)
        
        # Сохранение кода
        output_file_name = f"{file_name}+{sample_idx}.py"
        with open(output_dir / output_file_name, 'w') as f:
            f.write(py_string)
        
        results.append({
            'file_name': file_name,
            'code': py_string
        })
    
    return results


def run_cad_recode_inference(data_path, split, model_path, output_dir,
                             device='cuda', n_samples=1, max_samples=50):
    """
    Запускает полный цикл инференса модели CAD-Recode на датасете.
    
    Выполняет загрузку модели, создание датасета, генерацию CadQuery кодов
    для всех образцов и сохранение результатов. Поддерживает множественные
    генерации для каждого образца (n_samples > 1).
    
    Процесс выполнения:
        1. Загрузка токенизатора и конфигурации модели
        2. Создание и загрузка модели CAD-Recode
        3. Создание датасета из STL файлов
        4. Генерация CadQuery кодов для каждого образца (возможно несколько раз)
        5. Сохранение результатов в указанную директорию
    
    Args:
        data_path (str): Путь к корневой директории с данными.
            Должна содержать поддиректорию с названием split.
        split (str): Название датасета для тестирования.
            Должна быть поддиректорией в data_path со STL файлами.
        model_path (str): Путь к директории с моделью CAD-Recode.
            Должна содержать config.json и файлы модели.
        output_dir (str): Директория для сохранения сгенерированных CadQuery файлов.
            Будет создана автоматически, если не существует.
        device (str, optional): Устройство для вычислений ('cuda' или 'cpu').
            По умолчанию 'cuda'.
        n_samples (int, optional): Количество генераций на каждый образец.
            Позволяет получить несколько вариантов кода для одного объекта.
            По умолчанию 1.
        max_samples (int, optional): Максимальное количество образцов для обработки.
            Если None, обрабатываются все найденные STL файлы.
            По умолчанию 50.
    
    Raises:
        FileNotFoundError: Если не найдены модель, данные или токенизатор
        RuntimeError: Если не удалось загрузить модель или выполнить генерацию
        ValueError: Если параметры имеют недопустимые значения
    
    Note:
        Модель загружается с использованием bfloat16 для экономии памяти GPU.
        Используется flash_attention_2 для ускорения вычислений на GPU.
        Результаты сохраняются в формате '{file_name}+{sample_idx}.py'.
    
    Example:
        >>> run_cad_recode_inference(
        ...     data_path='/workspace/data',
        ...     split='deepcad_test_mesh',
        ...     model_path='/workspace/models/cad-recode-v1.5',
        ...     output_dir='/workspace/results/cad_recode',
        ...     device='cuda',
        ...     n_samples=1,
        ...     max_samples=50
        ... )
    """
    print(f"Загрузка модели с {model_path}...")
    start_time = time.time()
    
    # Загрузка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        '/workspace/models/Qwen2-1.5B',
        pad_token='<|im_end|>',
        padding_side='left')
    
    print("Загрузка конфигурации модели...")
    config = Qwen2Config.from_pretrained(model_path)
    
    print("Создание и загрузка модели...")
    # Загрузка модели
    model = CADRecode.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device_map=device if device == 'cuda' else None,
        attn_implementation='flash_attention_2' if device == 'cuda' else None)
    
    model.eval()
    if device == 'cuda':
        model = model.to(device)
    
    print(f"Модель загружена за {time.time() - start_time:.2f} секунд")
    
    # Создание директории для результатов
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Загрузка датасета
    dataset = STLDataset(data_path, split, max_samples)
    # Используем batch_size=8 для оптимальной генерации последовательностей
    # num_workers=20 для использования всех ядер процессора (20 потоков)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=20)
    print(f"Найдено {len(dataset)} STL файлов для обработки")
    
    # Генерация CadQuery кодов для каждого объекта
    for sample_idx in range(n_samples):
        print(f"\nГенерация варианта {sample_idx+1}/{n_samples}...")
        
        for batch in tqdm(dataloader, desc=f"Обработка файлов (вариант {sample_idx+1})"):
            run_inference_batch(
                model,
                tokenizer,
                batch,
                output_dir,
                device,
                sample_idx
            )
    
    print(f"\n✅ Генерация CadQuery кодов завершена. Результаты сохранены в {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Тестирование модели CAD-Recode')
    parser.add_argument('--data-path', type=str, default='/workspace/data',
                        help='Путь к директории с данными')
    parser.add_argument('--split', type=str, default='deepcad_test_mesh',
                        help='Название датасета для тестирования')
    parser.add_argument('--model-path', type=str, default='/workspace/models/cad-recode-v1.5',
                        help='Путь к модели CAD-Recode')
    parser.add_argument('--output-dir', type=str, default='/workspace/work_dirs/cad_recode_results',
                        help='Директория для сохранения результатов')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство для вычислений (cuda/cpu)')
    parser.add_argument('--n-samples', type=int, default=1,
                        help='Количество генераций на объект')
    parser.add_argument('--max-samples', type=int, default=50,
                        help='Максимальное количество объектов для обработки')
    
    args = parser.parse_args()
    
    # Проверка доступности GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA недоступна, переключаемся на CPU")
        args.device = 'cpu'
    
    if torch.cuda.is_available():
        print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Загрузка конфигурации для проверки
    print("Проверка конфигурации модели...")
    try:
        config = Qwen2Config.from_pretrained(args.model_path)
        print(f"✅ Конфигурация успешно загружена: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        print("Попытка загрузить конфигурацию из базовой модели...")
        config = Qwen2Config.from_pretrained('/workspace/models/Qwen2-1.5B')
    
    print("\n" + "="*50)
    print("ПАРАМЕТРЫ ЗАПУСКА:")
    print(f"  Данные: {args.data_path}/{args.split}")
    print(f"  Модель: {args.model_path}")
    print(f"  Результаты: {args.output_dir}")
    print(f"  Устройство: {args.device}")
    print(f"  Образцов: {args.max_samples} файлов × {args.n_samples} вариантов")
    print("="*50 + "\n")
    
    run_cad_recode_inference(
        args.data_path,
        args.split,
        args.model_path,
        args.output_dir,
        args.device,
        args.n_samples,
        args.max_samples
    )
