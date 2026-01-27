#!/usr/bin/env python3
"""
Модуль с реализацией модели Cadrille для генерации CadQuery кодов.

Cadrille - это мультимодальная модель на базе Qwen2VL, расширенная для работы
с точечными облаками, изображениями, видео и текстовыми описаниями для генерации
CadQuery кодов, описывающих 3D геометрию.

Основные компоненты:
    - Cadrille: основная модель, расширяющая Qwen2VLForConditionalGeneration
    - FourierPointEncoder: энкодер для точечных облаков с использованием Fourier features
    - FourierEmbedder: базовый класс для создания Fourier embeddings
    - collate: функция для подготовки батчей данных
    - find_assistant_content_sublist_indexes: вспомогательная функция для поиска ответов

Архитектура:
    - Базовый трансформер: Qwen2VL (поддержка мультимодальных входов)
    - Point encoder: FourierPointEncoder для кодирования точечных облаков
    - Visual encoder: встроенный encoder Qwen2VL для изображений и видео
    - Language model head: генерация токенов CadQuery кода

Используемые библиотеки:
    - torch: для работы с тензорами и нейронными сетями
    - transformers: для базовой модели Qwen2VL
    - qwen_vl_utils: для обработки визуальной информации

Пример использования:
    >>> from cadrille import Cadrille, collate
    >>> model = Cadrille.from_pretrained('maksimko123/cadrille')
    >>> # Использование модели для генерации...
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast


def collate(batch, processor, n_points, eval=False):
    """
    Подготавливает батч данных для модели Cadrille.
    
    Обрабатывает батч элементов датасета и преобразует их в формат,
    пригодный для передачи в модель. Поддерживает различные типы входных
    данных: точечные облака, изображения/видео и текстовые описания.
    
    Алгоритм:
        1. Определение типа входных данных для каждого элемента (PC/IMG/TEXT)
        2. Формирование сообщений в формате chat template
        3. Обработка визуальной информации (изображения/видео)
        4. Подготовка точечных облаков и масок
        5. Создание labels для обучения (только в режиме train)
    
    Args:
        batch (list): Список элементов датасета. Каждый элемент может содержать:
            - 'point_cloud' (np.ndarray, optional): Точечное облако формы (n_points, 3)
            - 'video' (list, optional): Список изображений PIL.Image
            - 'description' (str): Текстовое описание задачи
            - 'answer' (str, optional): Целевой CadQuery код (для обучения)
            - 'file_name' (str): Имя файла
        processor (AutoProcessor): Процессор для токенизации и обработки данных.
            Должен быть экземпляром Qwen2VLProcessor.
        n_points (int): Количество точек в точечном облаке. Используется для
            создания placeholder токенов для point cloud embeddings.
        eval (bool, optional): Режим работы. Если True, не создаются labels для обучения.
            По умолчанию False.
    
    Returns:
        dict: Словарь с подготовленными данными для модели, содержащий:
            - 'input_ids' (torch.Tensor): Токенизированные входные данные
            - 'attention_mask' (torch.Tensor): Маска внимания
            - 'pixel_values' (torch.Tensor, optional): Значения пикселей изображений
            - 'pixel_values_videos' (torch.Tensor, optional): Значения пикселей видео
            - 'video_grid_thw' (torch.Tensor, optional): Размеры сетки видео
            - 'point_clouds' (torch.Tensor): Тензор точечных облаков формы (batch, n_points, 3)
            - 'is_pc' (torch.Tensor): Булев тензор, указывающий элементы с точечными облаками
            - 'is_img' (torch.Tensor): Булев тензор, указывающий элементы с изображениями
            - 'labels' (torch.Tensor, optional): Labels для обучения (только если eval=False)
            - 'file_name' (list, optional): Имена файлов (только если eval=True)
    
    Raises:
        ValueError: Если processor не поддерживает требуемый функционал
        RuntimeError: Если не удалось обработать визуальную информацию
    
    Note:
        Для элементов с точечными облаками в начало последовательности добавляются
        pad токены в количестве n_points. Эти токены будут заменены на point embeddings
        в методе forward модели.
    
    Example:
        >>> from transformers import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
        >>> batch = [
        ...     {'point_cloud': pc_array, 'description': 'Generate cadquery code'},
        ...     {'video': [img1, img2], 'description': 'Generate cadquery code'}
        ... ]
        >>> inputs = collate(batch, processor, n_points=256, eval=True)
        >>> print(inputs.keys())
        dict_keys(['input_ids', 'attention_mask', 'point_clouds', 'is_pc', 'is_img', ...])
    """
    messages = []
    is_pc = [0] * len(batch)
    is_img = [0] * len(batch)
    
    # Формируем сообщения для каждого элемента батча в зависимости от режима (train/eval)
    if not eval:
        # Режим обучения: создаем полные диалоги с ответами ассистента
        for i, m in enumerate(batch):
            if 'video' in m.keys():
                # Элемент с изображениями/видео
                is_img[i] = 1
                message = [{
                        'role': 'user',
                        'content': [
                            {'type': 'video', 'video': m['video'], 'fps': 1.0},
                            {'type': 'text', 'text': m['description']}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': m['answer']}
                        ]
                    }]
            else:
                # Элемент с точечным облаком или текстом
                if 'point_cloud' in m.keys():
                    is_pc[i] = 1
                message = [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': m['description']}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': m['answer']}
                        ]
                    }]
            messages.append(message)
        # Применяем chat template без промпта генерации (для обучения)
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
            for msg in messages] 
    else:
        # Режим инференса: создаем только запросы пользователя без ответов
        for i, m in enumerate(batch):
            if 'video' in m.keys():
                is_img[i] = 1
                message = [{
                        'role': 'user',
                        'content': [
                            {'type': 'video', 'video': m['video'], 'fps': 1.0},
                            {'type': 'text', 'text': m['description']}
                        ]
                    }]
            else:
                if 'point_cloud' in m.keys():
                    is_pc[i] = 1
                message = [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': m['description']}
                        ]
                    }]
            messages.append(message)
        # Применяем chat template с промптом генерации (для инференса)
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
            for msg in messages]


    # Для элементов с точечными облаками добавляем pad токены в начало последовательности
    # Эти токены будут заменены на point embeddings в методе forward модели
    points_inputs = ''.join(n_points * [processor.tokenizer.pad_token])
    
    for i in range(len(texts)):
        if is_pc[i]:
            # Добавляем pad токены перед текстом для резервирования места под point embeddings
            texts[i] = points_inputs + texts[i]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt')

    inputs['point_clouds'] = torch.stack([
        torch.tensor(m['point_cloud']) if is_pc[i] else torch.zeros(n_points, 3)
        for i, m in enumerate(batch)])
    inputs['is_pc'] = torch.tensor(is_pc, dtype=torch.bool)
    inputs['is_img'] = torch.tensor(is_img, dtype=torch.bool)

    if 'pixel_values_videos' in inputs.keys():
        pixel_values_videos = inputs['pixel_values_videos'].new_zeros((
            len(batch), torch.prod(inputs['video_grid_thw'][0]),
            inputs['pixel_values_videos'].shape[1]))
        pixel_values_videos[inputs['is_img']] = torch.stack(
            torch.chunk(inputs['pixel_values_videos'], 
            chunks=sum(inputs['is_img'])))
        inputs['pixel_values_videos'] = pixel_values_videos
    
        video_grid_thw = inputs['video_grid_thw'].new_zeros((len(batch), 3))
        video_grid_thw[inputs['is_img']] = inputs['video_grid_thw']
        inputs['video_grid_thw'] = video_grid_thw

    if not eval:
        input_ids_lists = inputs['input_ids'].tolist()
        assert len(messages) == len(input_ids_lists)

        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1] = \
                    ids_list[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1]
            labels_list.append(label_ids)
        labels_ids = torch.tensor(labels_list, dtype=torch.int64)
        inputs['labels'] = labels_ids
    else:
        inputs['file_name'] = [m['file_name'] for m in batch]
    return inputs


def find_assistant_content_sublist_indexes(l):
    """
    Находит индексы начала и конца содержимого ассистента в токенизированной последовательности.
    
    Используется для создания labels при обучении модели. Ищет специальные токены,
    которые маркируют начало и конец ответа ассистента в chat template.
    
    Алгоритм:
        Ищет пары токенов (151644, 77091) как маркеры начала ответа и следующий
        токен 151645 как маркер конца ответа.
    
    Args:
        l (list): Список токенов (token IDs) из токенизированной последовательности.
            Должен содержать специальные токены для разметки ответов ассистента.
    
    Returns:
        list: Список кортежей (start_index, end_index) для каждого найденного
            ответа ассистента. Каждый кортеж содержит индексы начала и конца
            содержимого ответа в последовательности.
    
    Note:
        Токены 151644 и 77091 соответствуют специальным токенам в токенизаторе
        Qwen2VL для маркировки ответов ассистента. Токен 151645 - маркер конца ответа.
    
    Example:
        >>> tokens = [1, 2, 151644, 77091, 10, 20, 30, 151645, 4, 5]
        >>> indexes = find_assistant_content_sublist_indexes(tokens)
        >>> print(indexes)
        [(2, 7)]  # Ответ начинается с индекса 2 и заканчивается на индексе 7
    """
    start_indexes = []
    end_indexes = []

    # Проходим по последовательности токенов и ищем маркеры ответов ассистента
    # Токены 151644 и 77091 вместе образуют маркер начала ответа ассистента
    for i in range(len(l) - 1):
        # Проверяем, является ли текущая пара токенов маркером начала ответа
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            # Ищем соответствующий маркер конца ответа (токен 151645)
            # после найденного начала
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break  # Переходим к следующему началу после нахождения конца

    # Возвращаем список кортежей (начало, конец) для каждого найденного ответа
    return list(zip(start_indexes, end_indexes))


class FourierEmbedder(nn.Module):
    """
    Базовый класс для создания Fourier positional encodings.
    
    Использует синусоидальные функции различных частот для преобразования
    координат в высокоразмерные embeddings. Это позволяет модели лучше
    понимать пространственные отношения в точечных облаках.
    
    Fourier features улучшают способность нейронных сетей аппроксимировать
    высокочастотные функции, что важно для точного представления геометрии.
    
    Attributes:
        frequencies (torch.Tensor): Буфер с частотами для Fourier преобразования.
            Форма: (num_freqs,)
        include_input (bool): Включать ли исходные координаты в выходной тензор.
    
    Note:
        Использует register_buffer для сохранения frequencies как части состояния
        модели без необходимости обновления через оптимизатор.
    """
    
    def __init__(self,
                 num_freqs=6,
                 logspace=True,
                 include_input=True,
                 include_pi=True):
        """
        Инициализирует FourierEmbedder.
        
        Args:
            num_freqs (int, optional): Количество частот для Fourier преобразования.
                Больше частот дают более детальное представление, но увеличивают
                размерность выхода. По умолчанию 6.
            logspace (bool, optional): Использовать ли логарифмический масштаб частот.
                Если True, частоты будут 2^0, 2^1, ..., 2^(num_freqs-1).
                Если False, равномерное распределение. По умолчанию True.
            include_input (bool, optional): Включать ли исходные координаты в выход.
                По умолчанию True.
            include_pi (bool, optional): Умножать ли частоты на π.
                По умолчанию True.
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32)

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer('frequencies', frequencies, persistent=False)
        self.include_input = include_input

    def forward(self, x):
        """
        Применяет Fourier преобразование к входным координатам.
        
        Args:
            x (torch.Tensor): Входные координаты формы (..., 3).
                Последняя размерность должна быть 3 (x, y, z координаты).
        
        Returns:
            torch.Tensor: Fourier embeddings формы (..., out_dim), где out_dim
                зависит от num_freqs и include_input. Содержит исходные координаты
                (если include_input=True) и их синусоидальные преобразования.
        """
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class FourierPointEncoder(nn.Module):
    """
    Энкодер точечных облаков с использованием Fourier features.
    
    Преобразует координаты точек из точечного облака в embeddings для модели.
    Использует FourierEmbedder для создания позиционных кодирований и линейный
    слой для проекции в пространство скрытых состояний модели.
    
    Архитектура:
        1. FourierEmbedder: преобразует координаты (x, y, z) в Fourier features
        2. Linear projection: проекция в hidden_size измерений
    
    Attributes:
        fourier_embedder (FourierEmbedder): Энкодер для создания Fourier features
        projection (nn.Linear): Линейный слой для проекции в hidden_size
    
    Note:
        Используется только для кодирования первых 3 координат точек (x, y, z).
        Если точки содержат дополнительные признаки, они игнорируются.
    """
    
    def __init__(self, hidden_size):
        """
        Инициализирует FourierPointEncoder.
        
        Args:
            hidden_size (int): Размерность скрытого пространства модели.
                Должен совпадать с hidden_size конфигурации модели.
        """
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points):
        """
        Кодирует точечное облако в embeddings.
        
        Args:
            points (torch.Tensor): Точечное облако формы (batch, n_points, 3) или
                (batch, n_points, features). Используются только первые 3 координаты.
        
        Returns:
            torch.Tensor: Embeddings формы (batch, n_points, hidden_size).
                Каждая точка представлена вектором размерности hidden_size.
        """
        # Используем только первые 3 координаты (x, y, z) для Fourier encoding
        x = self.fourier_embedder(points[..., :3])
        x = self.projection(x)
        return x
    

class Cadrille(Qwen2VLForConditionalGeneration):
    """
    Модель Cadrille для генерации CadQuery кодов из различных входных данных.
    
    Расширяет Qwen2VLForConditionalGeneration для поддержки:
    - Точечных облаков (point clouds) через FourierPointEncoder
    - Изображений через встроенный visual encoder Qwen2VL
    - Видео через встроенный visual encoder Qwen2VL
    - Текстовых описаний через текстовый encoder
    
    Архитектура:
        - Базовый трансформер: Qwen2VL (мультимодальная модель)
        - Point encoder: FourierPointEncoder для кодирования точечных облаков
        - Visual encoder: встроенный encoder Qwen2VL для изображений и видео
        - Language model head: генерация токенов CadQuery кода
    
    Процесс работы:
        1. Входные данные (текст/изображение/точечное облако) обрабатываются соответствующими энкодерами
        2. Embeddings объединяются в единую последовательность
        3. Трансформер обрабатывает последовательность
        4. Language model head генерирует токены CadQuery кода
    
    Attributes:
        point_encoder (FourierPointEncoder): Энкодер для точечных облаков.
            Создается в float32 для стабильности вычислений.
        model: Базовый Qwen2VL модель для обработки последовательностей.
        lm_head: Головка для генерации токенов (наследуется от Qwen2VL).
        visual: Visual encoder для изображений и видео (наследуется от Qwen2VL).
    
    Note:
        Point encoder создается в float32, но модель работает в bfloat16
        для экономии памяти GPU. Конвертация выполняется автоматически.
    
    Example:
        >>> from cadrille import Cadrille
        >>> model = Cadrille.from_pretrained('maksimko123/cadrille')
        >>> # Генерация кода из точечного облака
        >>> outputs = model.generate(
        ...     input_ids=input_ids,
        ...     point_clouds=point_clouds,
        ...     is_pc=is_pc,
        ...     max_new_tokens=768
        ... )
    """
    
    def __init__(self, config):
        """
        Инициализирует модель Cadrille.
        
        Args:
            config: Конфигурация модели (Qwen2VLConfig).
                Должна содержать параметры hidden_size, vocab_size и другие.
        
        Note:
            Point encoder создается в float32 для обеспечения численной стабильности
            при вычислении Fourier features, затем dtype возвращается к bfloat16.
        """
        super().__init__(config)
     
        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        point_clouds=None,
        is_pc=None,
        is_img=None):
        """
        Прямой проход модели Cadrille.
        
        Обрабатывает различные типы входных данных и генерирует логиты для
        предсказания следующего токена. Поддерживает кэширование для ускорения
        генерации.
        
        Алгоритм обработки:
            1. Создание текстовых embeddings из input_ids (если inputs_embeds не заданы)
            2. Обработка изображений через visual encoder (если pixel_values заданы)
            3. Обработка видео через visual encoder (если pixel_values_videos заданы)
            4. Встраивание point cloud embeddings в позиции pad токенов (только на первом проходе)
            5. Вычисление position_ids с учетом визуальных токенов (RoPE)
            6. Прямой проход через трансформер
            7. Вычисление логитов через language model head
            8. Вычисление loss (если labels заданы)
        
        Args:
            input_ids (torch.Tensor, optional): Токенизированные входные данные.
                Форма: (batch_size, seq_length). По умолчанию None.
            attention_mask (torch.Tensor, optional): Маска внимания.
                Форма: (batch_size, seq_length). По умолчанию None.
            position_ids (torch.Tensor, optional): Позиционные индексы для RoPE.
                Вычисляются автоматически, если не заданы. По умолчанию None.
            past_key_values (Cache, optional): Кэш ключей и значений для ускорения генерации.
                Используется для инкрементальной генерации. По умолчанию None.
            inputs_embeds (torch.Tensor, optional): Предвычисленные embeddings.
                Если заданы, input_ids игнорируются. По умолчанию None.
            labels (torch.Tensor, optional): Labels для вычисления loss.
                Форма: (batch_size, seq_length). Используется только при обучении.
                По умолчанию None.
            use_cache (bool, optional): Использовать ли кэширование ключей/значений.
                По умолчанию None (используется значение из config).
            output_attentions (bool, optional): Возвращать ли attention weights.
                По умолчанию None (используется значение из config).
            output_hidden_states (bool, optional): Возвращать ли скрытые состояния.
                По умолчанию None (используется значение из config).
            return_dict (bool, optional): Возвращать ли объект вместо tuple.
                По умолчанию None (используется значение из config).
            pixel_values (torch.Tensor, optional): Значения пикселей изображений.
                Форма: (batch_size, num_images, channels, height, width).
                По умолчанию None.
            pixel_values_videos (torch.Tensor, optional): Значения пикселей видео.
                Форма: (batch_size, num_frames, channels, height, width).
                По умолчанию None.
            image_grid_thw (torch.Tensor, optional): Размеры сетки изображений.
                Используется для правильной обработки нескольких изображений.
                По умолчанию None.
            video_grid_thw (torch.Tensor, optional): Размеры сетки видео.
                Используется для правильной обработки видео.
                По умолчанию None.
            rope_deltas (torch.Tensor, optional): Смещения для RoPE.
                Вычисляются автоматически. По умолчанию None.
            cache_position (torch.Tensor, optional): Позиции в кэше для инкрементальной генерации.
                По умолчанию None.
            point_clouds (torch.Tensor, optional): Точечные облака для кодирования.
                Форма: (batch_size, n_points, 3). По умолчанию None.
            is_pc (torch.Tensor, optional): Булев тензор, указывающий элементы с точечными облаками.
                Форма: (batch_size,). По умолчанию None.
            is_img (torch.Tensor, optional): Булев тензор, указывающий элементы с изображениями/видео.
                Форма: (batch_size,). По умолчанию None.
        
        Returns:
            Qwen2VLCausalLMOutputWithPast или tuple: Если return_dict=True, возвращает
                объект с полями:
                - loss (torch.Tensor, optional): Cross-entropy loss (если labels заданы)
                - logits (torch.Tensor): Логиты для предсказания токенов.
                    Форма: (batch_size, seq_length, vocab_size)
                - past_key_values (Cache, optional): Кэш для следующей итерации генерации
                - hidden_states (tuple, optional): Скрытые состояния всех слоев
                - attentions (tuple, optional): Attention weights всех слоев
                - rope_deltas (torch.Tensor): Смещения для RoPE
        
        Raises:
            ValueError: Если количество визуальных токенов не совпадает с количеством features
            RuntimeError: Если не удалось обработать входные данные
        
        Note:
            Point cloud embeddings встраиваются только на первом проходе (когда
            past_key_values пуст). На последующих итерациях генерации используются
            только текстовые токены.
        
        Example:
            >>> outputs = model.forward(
            ...     input_ids=input_ids,
            ...     attention_mask=attention_mask,
            ...     point_clouds=point_clouds,
            ...     is_pc=is_pc,
            ...     is_img=is_img
            ... )
            >>> logits = outputs.logits
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if is_img.sum() > 0 and pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos[is_img]
                pixel_values_videos = pixel_values_videos.view(-1, pixel_values_videos.shape[-1])
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_grid_thw = video_grid_thw[is_img]
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Встраивание точечных облаков: заменяем pad токены на point embeddings
            # Выполняется только на первом проходе (когда нет past_key_values)
            # На последующих итерациях генерации используются только текстовые токены из кэша
            if is_pc.sum() > 0 and (past_key_values is None or past_key_values.get_seq_length() == 0):
                # Кодируем точечные облака через FourierPointEncoder
                # Используем float32 для стабильности вычислений Fourier features
                point_embeds = self.point_encoder(point_clouds.float()).bfloat16()
                
                # Находим позиции начала последовательностей (где находятся pad токены)
                # attention_mask.sum(axis=1) дает количество реальных токенов в каждой последовательности
                # Разница с полной длиной дает количество pad токенов в начале
                start_idxs = attention_mask.shape[1] - attention_mask.sum(axis=1)
                
                # Встраиваем point embeddings в позиции pad токенов для каждого элемента батча
                # Заменяем embeddings pad токенов на point embeddings
                for i, start_idx in enumerate(start_idxs):
                    if is_pc[i]:
                        # Заменяем embeddings pad токенов на point embeddings
                        inputs_embeds[i, start_idx:start_idx + point_embeds.shape[1], :] = point_embeds[i]

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Подготавливает входные данные для генерации.
        
        Расширяет базовый метод prepare_inputs_for_generation для передачи
        дополнительных параметров (point_clouds, is_pc, is_img) через все
        итерации генерации.
        
        Args:
            *args: Позиционные аргументы базового метода
            **kwargs: Именованные аргументы, должны содержать:
                - point_clouds (torch.Tensor): Точечные облака
                - is_pc (torch.Tensor): Маска для точечных облаков
                - is_img (torch.Tensor): Маска для изображений
        
        Returns:
            dict: Подготовленные входные данные с добавленными полями:
                - point_clouds: Точечные облака
                - is_pc: Маска для точечных облаков
                - is_img: Маска для изображений
        """
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs['point_clouds'] = kwargs['point_clouds']
        model_inputs['is_pc'] = kwargs['is_pc']
        model_inputs['is_img'] = kwargs['is_img']
        return model_inputs
