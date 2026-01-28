# Сравнение архитектур CAD-Recode и Cadrille

## Оглавление

1. [Базовые модели Qwen](#базовые-модели-qwen)
2. [Модификации базовых моделей](#модификации-базовых-моделей)
3. [Детальное сравнение обработки облаков точек](#детальное-сравнение-обработки-облаков-точек)
4. [Обработка мультимодальных входов в Cadrille](#обработка-мультимодальных-входов-в-cadrille)
5. [Сравнительная таблица архитектур](#сравнительная-таблица-архитектур)

---

## Базовые модели Qwen

### Qwen2ForCausalLM (база для CAD-Recode)

**Характеристики:**
- **Тип:** Чистая языковая модель (text-only)
- **Архитектура:** Transformer decoder-only
- **Назначение:** Генерация текста на основе текстового контекста
- **Модальности:** Только текст

**Ключевые компоненты:**
- `Qwen2Model`: Базовый трансформер для обработки последовательностей токенов
- `lm_head`: Линейный слой для генерации токенов из скрытых состояний
- `embed_tokens`: Эмбеддинги токенов
- Стандартная RoPE (Rotary Position Embedding) для позиционного кодирования

**Конфигурация (Qwen2Config):**
- `hidden_size`: Размерность скрытого пространства (обычно 1536 для моделей 1.5B)
- `vocab_size`: Размер словаря токенизатора
- `num_layers`: Количество трансформерных слоев
- `num_attention_heads`: Количество голов внимания
- Нет параметров для визуальных входов

### Qwen2VLForConditionalGeneration (база для Cadrille)

**Характеристики:**
- **Тип:** Мультимодальная модель (text + vision)
- **Архитектура:** Transformer decoder-only с visual encoder
- **Назначение:** Генерация текста на основе текста, изображений и видео
- **Модальности:** Текст, изображения, видео

**Ключевые компоненты:**
- `Qwen2VLModel`: Базовый трансформер с поддержкой визуальных входов
- `visual`: Visual encoder для обработки изображений и видео
- `lm_head`: Линейный слой для генерации токенов
- `embed_tokens`: Эмбеддинги токенов
- Расширенная RoPE с учетом визуальных токенов

**Конфигурация (Qwen2VLConfig):**
- Наследует все параметры от Qwen2Config
- Дополнительные параметры:
  - `image_token_id`: ID токена для изображений
  - `video_token_id`: ID токена для видео
  - Параметры visual encoder (размеры патчей, количество визуальных токенов)

**Визуальный энкодер:**
- Обрабатывает изображения и видео в патчи
- Преобразует визуальные данные в последовательность токенов
- Интегрируется с текстовыми токенами через специальные токены-маркеры

### Ключевые различия

| Аспект | Qwen2ForCausalLM | Qwen2VLForConditionalGeneration |
|--------|-------------------|----------------------------------|
| Модальности | Только текст | Текст + изображения + видео |
| Visual encoder | Нет | Да (встроенный) |
| Специальные токены | Стандартные | image_token_id, video_token_id |
| RoPE | Стандартная | С учетом визуальных токенов |
| Обработка последовательностей | Только текстовые токены | Текстовые + визуальные токены |
| Размер модели | Меньше | Больше (из-за visual encoder) |

---

## Модификации базовых моделей

### CAD-Recode: Расширение Qwen2ForCausalLM

**Наследование:**
```python
class CADRecode(Qwen2ForCausalLM):
    config_class = Qwen2Config
```

**Добавленные компоненты:**

1. **FourierPointEncoder** (`test_cad_recode.py`, строки 27-87):
   - Прямое вычисление Fourier features без базового класса
   - Параметры: `num_freqs=8`, `include_pi=False`
   - Выходная размерность: 51 (3 исходных + 8*2*3 Fourier features)
   - Проекция: `nn.Linear(51, hidden_size)`

2. **Модификация `__init__`:**
   - Вызов `PreTrainedModel.__init__()` вместо `super().__init__()`
   - Создание `Qwen2Model` напрямую
   - Создание `lm_head` вручную
   - Создание `point_encoder` в float32 для стабильности

3. **Модификация `forward`:**
   - Добавлена обработка `point_cloud` параметра
   - Встраивание point embeddings через `attention_mask == -1`
   - Обновление attention_mask после встраивания

**Изменения в архитектуре:**
- Добавлен один новый компонент: `FourierPointEncoder`
- Модифицирован метод `forward` для интеграции point embeddings
- Базовый трансформер используется без изменений

### Cadrille: Расширение Qwen2VLForConditionalGeneration

**Наследование:**
```python
class Cadrille(Qwen2VLForConditionalGeneration):
```

**Добавленные компоненты:**

1. **FourierEmbedder** (`cadrille.py`, строки 279-354):
   - Базовый класс для создания Fourier positional encodings
   - Параметры: `num_freqs=6` (по умолчанию), `logspace=True`, `include_input=True`, `include_pi=True`
   - Используется как базовый класс для `FourierPointEncoder`

2. **FourierPointEncoder** (`cadrille.py`, строки 357-405):
   - Использует `FourierEmbedder` как базовый класс
   - Параметры: `num_freqs=8`, `include_pi=False`
   - Выходная размерность: 51 (аналогично CAD-Recode)
   - Проекция: `nn.Linear(51, hidden_size)`

3. **Модификация `__init__`:**
   - Вызов `super().__init__(config)` для наследования всей функциональности Qwen2VL
   - Создание `point_encoder` в float32, затем возврат к bfloat16

4. **Модификация `forward`:**
   - Добавлена обработка `point_clouds`, `is_pc`, `is_img`
   - Интеграция с существующей обработкой изображений/видео
   - Встраивание point embeddings через динамические индексы
   - RoPE с учетом всех модальностей

**Изменения в архитектуре:**
- Добавлены два компонента: `FourierEmbedder` и `FourierPointEncoder`
- Модифицирован метод `forward` для интеграции point embeddings с визуальными токенами
- Используется вся функциональность базовой модели Qwen2VL

---

## Детальное сравнение обработки облаков точек

### Реализация Fourier Point Encoding

#### CAD-Recode (`test_cad_recode.py`, строки 27-87)

```python
class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)
    
    def forward(self, points):
        x = points
        # Создаем Fourier features: умножаем координаты на частоты
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        # Объединяем исходные координаты с sin/cos преобразованиями
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        # Проектируем в пространство скрытых состояний
        x = self.projection(x)
        return x
```

**Особенности:**
- Прямое вычисление без базового класса
- Частоты: `2^0, 2^1, ..., 2^7` (8 частот)
- Формула: `cat([points, sin(freqs * points), cos(freqs * points)])`
- Выходная размерность перед проекцией: 51 (3 + 8*2*3)

#### Cadrille (`cadrille.py`, строки 279-405)

```python
class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=6, logspace=True, include_input=True, include_pi=True):
        # ... инициализация frequencies ...
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.include_input = include_input
    
    def forward(self, x):
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)

class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=False)
        self.projection = nn.Linear(51, hidden_size)
    
    def forward(self, points):
        x = self.fourier_embedder(points[..., :3])
        x = self.projection(x)
        return x
```

**Особенности:**
- Использует базовый класс `FourierEmbedder`
- Частоты: `2^0, 2^1, ..., 2^7` (8 частот, `include_pi=False`)
- Та же формула через класс: `cat([points, sin(freqs * points), cos(freqs * points)])`
- Выходная размерность перед проекцией: 51 (аналогично CAD-Recode)

**Сравнение реализаций:**
- **Идентичность формул:** Обе используют одинаковую формулу Fourier encoding
- **Различия в структуре:** Cadrille использует базовый класс, CAD-Recode - прямое вычисление
- **Параметры:** Одинаковые (`num_freqs=8`, `include_pi=False`)
- **Проекция:** Одинаковая (`nn.Linear(51, hidden_size)`)

### Интеграция Point Embeddings в последовательность

#### CAD-Recode (`test_cad_recode.py`, строки 238-265)

**Процесс:**

1. **Подготовка входных данных** (`run_inference_batch()`, строка 607):
   ```python
   input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
   attention_mask = [-1] * len(point_cloud) + [1]
   ```

2. **В forward методе** (строки 238-265):
   ```python
   # Шаг 1: Создание текстовых embeddings
   inputs_embeds = self.model.embed_tokens(input_ids)
   
   # Шаг 2: Кодирование point cloud
   point_embeds = self.point_encoder(point_cloud).to(inputs_embeds.dtype)
   
   # Шаг 3: Замена pad токенов на point embeddings
   inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
   
   # Шаг 4: Обновление attention_mask
   attention_mask = attention_mask.clone()
   attention_mask[attention_mask == -1] = 1
   ```

**Особенности:**
- Использует специальное значение `-1` в `attention_mask` для маркировки позиций PC
- Векторная операция через булевую маску: `inputs_embeds[attention_mask == -1]`
- Pad токены создаются явно перед вызовом модели

#### Cadrille (`cadrille.py`, строки 632-650)

**Процесс:**

1. **Подготовка входных данных** (`collate()`, строка 182):
   ```python
   # Добавление pad токенов перед текстом для резервирования места под point embeddings
   points_inputs = ''.join(n_points * [processor.tokenizer.pad_token])
   if is_pc[i]:
       texts[i] = points_inputs + texts[i]
   ```

2. **В forward методе** (строки 632-650):
   ```python
   # Шаг 1: Создание текстовых embeddings
   inputs_embeds = self.model.embed_tokens(input_ids)
   
   # Шаг 2: Кодирование point cloud
   point_embeds = self.point_encoder(point_clouds.float()).bfloat16()
   
   # Шаг 3: Вычисление позиций начала последовательностей
   start_idxs = attention_mask.shape[1] - attention_mask.sum(axis=1)
   
   # Шаг 4: Встраивание через цикл по батчу
   for i, start_idx in enumerate(start_idxs):
       if is_pc[i]:
           inputs_embeds[i, start_idx:start_idx + point_embeds.shape[1], :] = point_embeds[i]
   ```

**Особенности:**
- Позиции определяются динамически через `attention_mask.sum(axis=1)`
- Встраивание через цикл по элементам батча
- Pad токены добавляются в `collate()` перед применением chat template

**Ключевые различия:**

| Аспект | CAD-Recode | Cadrille |
|--------|------------|----------|
| Маркировка позиций | `attention_mask == -1` | Динамический расчет через `start_idxs` |
| Метод встраивания | Векторная операция через маску | Цикл по батчу |
| Подготовка данных | Явное создание pad токенов | Добавление в `collate()` |
| Обработка батча | Векторизованная операция | Поэлементная обработка |

### Обработка на последующих итерациях генерации

**Обе модели:**

- Point embeddings встраиваются **только на первом проходе**
- Условие: `past_key_values is None` или `past_key_values.get_seq_length() == 0`
- На последующих итерациях используются только текстовые токены из кэша
- Это оптимизация для ускорения генерации: point embeddings вычисляются один раз

---

## Обработка мультимодальных входов в Cadrille

### 1. Обработка текста

**Файлы:** `cadrille.py`: функция `collate()` (строки 130-141, 162-167)

**Процесс:**

1. **Формирование сообщений:**
   ```python
   message = [{
       'role': 'user',
       'content': [
           {'type': 'text', 'text': m['description']}
       ]
   }]
   ```

2. **Применение chat template:**
   - Режим train: `processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)`
   - Режим eval: `processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)`

3. **Токенизация:**
   ```python
   inputs = processor(
       text=texts,
       images=image_inputs,
       videos=video_inputs,
       padding=True,
       return_tensors='pt'
   )
   ```

4. **Интеграция в forward:**
   - Текстовые токены обрабатываются через `embed_tokens()` (строка 592)
   - Стандартная обработка через трансформер

**Особенности:**
- Использует полноценный chat template от Qwen2VL
- Поддержка диалоговой структуры (user/assistant)
- Автоматическая обработка специальных токенов

### 2. Обработка изображений

**Файлы:** `cadrille.py`: `forward()` (строки 593-609), `collate()` (строки 184-190)

**Процесс:**

1. **Подготовка в collate():**
   ```python
   image_inputs, video_inputs = process_vision_info(messages)
   inputs = processor(..., images=image_inputs, ...)
   ```

2. **Обработка в forward()** (строки 593-609):
   ```python
   if pixel_values is not None:
       # Конвертация dtype
       pixel_values = pixel_values.type(self.visual.get_dtype())
       
       # Обработка через visual encoder
       image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
       
       # Проверка соответствия токенов и features
       n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
       n_image_features = image_embeds.shape[0]
       if n_image_tokens != n_image_features:
           raise ValueError(...)
       
       # Встраивание через masked_scatter
       image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
       inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
   ```

**Особенности:**
- Использует встроенный visual encoder от Qwen2VL
- Проверка соответствия количества токенов и features
- Встраивание через `masked_scatter()` для замены image_token_id на embeddings

### 3. Обработка "видео" (множественных изображений с разных ракурсов)

**ВАЖНО:** В Cadrille "video" означает список изображений одного 3D объекта с разных ракурсов, а не настоящее видео.

**Файлы:**
- `dataset.py`: `get_img()` (строки 378-445) - рендеринг с 4 ракурсов
- `cadrille.py`: `collate()` (строки 110-125, 150-158), `forward()` (строки 611-630)

**Процесс рендеринга** (`dataset.py`, строки 378-445):

1. **Загрузка и нормализация mesh:**
   ```python
   mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
   mesh.apply_transform(trimesh.transformations.scale_matrix(1 / self.normalize_std_img))
   mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
   ```

2. **Рендеринг с 4 ракурсов:**
   ```python
   fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
   images = []
   for front in fronts:
       image = mesh_to_image(mesh, camera_distance=-0.9, front=front, img_size=self.img_size)
       images.append(image)
   ```

3. **Объединение изображений:**
   - `num_imgs=1`: одно изображение с ракурса [1,1,1]
   - `num_imgs=2`: горизонтальная композиция двух изображений
   - `num_imgs=4`: сетка 2x2 с четырьмя ракурсами

4. **Передача в модель:**
   ```python
   input_item = {
       'video': images,  # Список PIL.Image
       'description': 'Generate cadquery code'
   }
   ```

**Обработка в модели** (`cadrille.py`, строки 611-630):

```python
if is_img.sum() > 0 and pixel_values_videos is not None:
    # Обработка только элементов с изображениями
    pixel_values_videos = pixel_values_videos[is_img]
    pixel_values_videos = pixel_values_videos.view(-1, pixel_values_videos.shape[-1])
    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
    
    # Обработка через visual encoder с video_grid_thw
    video_grid_thw = video_grid_thw[is_img]
    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
    
    # Проверка соответствия токенов
    n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
    n_video_features = video_embeds.shape[0]
    if n_video_tokens != n_video_features:
        raise ValueError(...)
    
    # Встраивание через masked_scatter
    video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
```

**Особенности:**
- Используется `video_token_id` для маркировки позиций
- `video_grid_thw` определяет структуру сетки изображений
- `fps=1.0` указывает на статичность (не настоящее видео)
- Представляет 3D геометрию через 2D проекции с разных ракурсов

### 4. Обработка точечных облаков (PC)

См. раздел [Детальное сравнение обработки облаков точек](#детальное-сравнение-обработки-облаков-точек) выше.

В Cadrille PC обрабатывается аналогично CAD-Recode, но в контексте мультимодальной модели, где могут присутствовать также изображения и текст.

### 5. RoPE с визуальными токенами

**Файлы:** `cadrille.py`: `forward()` (строки 656-677)

**Процесс:**

```python
if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
    # Вычисление RoPE index с учетом визуальных токенов
    if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
        position_ids, rope_deltas = self.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask
        )
        self.rope_deltas = rope_deltas
    else:
        # Использование предвычисленных rope_deltas
        ...
```

**Особенности:**
- Метод `get_rope_index()` учитывает визуальные токены
- `rope_deltas` корректируют позиционное кодирование для визуальных токенов
- Вычисление `position_ids` с учетом `image_grid_thw` и `video_grid_thw`
- Это необходимо для правильной обработки последовательностей с визуальными токенами

---

## Сравнительная таблица архитектур

| Аспект | CAD-Recode | Cadrille |
|--------|------------|----------|
| **Базовая модель** | Qwen2ForCausalLM | Qwen2VLForConditionalGeneration |
| **Поддержка модальностей** | **Только PC** | Текст + PC + Изображения + Множественные изображения (видео) |
| **Visual encoder** | Нет | Да (наследуется от Qwen2VL) |
| **Fourier encoding PC** | Прямое вычисление | Через FourierEmbedder класс |
| **Обработка PC** | Через attention_mask == -1 | Через start_idxs из attention_mask |
| **Интеграция PC** | Векторная операция через маску | Цикл по батчу с динамическими индексами |
| **Обработка текста** | Только промпт | Полноценный chat template |
| **Обработка изображений** | Нет | Через visual encoder + image_token_id |
| **Обработка "видео"** | Нет | Множественные изображения с разных ракурсов через visual encoder |
| **RoPE обработка** | Стандартная | С учетом визуальных токенов (image/video) |
| **Специальные токены** | Стандартные | image_token_id, video_token_id |
| **Размер модели** | Меньше | Больше (из-за visual encoder) |
| **Сложность** | Проще | Сложнее (мультимодальность) |

---

## Выводы

### Архитектурные решения

1. **CAD-Recode:**
   - Выбрана простая архитектура на базе Qwen2ForCausalLM
   - Фокус только на обработке точечных облаков
   - Минимальные модификации базовой модели
   - Оптимизирована для работы только с PC

2. **Cadrille:**
   - Выбрана мультимодальная архитектура на базе Qwen2VL
   - Поддержка множественных модальностей (текст, PC, изображения)
   - Использование всей функциональности базовой модели
   - Более гибкая, но более сложная архитектура

### Преимущества и компромиссы

**CAD-Recode:**
- ✅ Проще в реализации и понимании
- ✅ Меньше параметров модели
- ✅ Быстрее инференс (только PC)
- ❌ Ограничен только точечными облаками
- ❌ Нет поддержки других модальностей

**Cadrille:**
- ✅ Поддержка множественных модальностей
- ✅ Более гибкая архитектура
- ✅ Использование визуальной информации
- ❌ Больше параметров модели
- ❌ Сложнее в реализации
- ❌ Медленнее инференс (обработка визуальных данных)

### Рекомендации по использованию

- **Используйте CAD-Recode**, если:
  - У вас есть только точечные облака
  - Нужна максимальная скорость инференса
  - Требуется простая архитектура

- **Используйте Cadrille**, если:
  - У вас есть доступ к изображениям или текстовым описаниям
  - Нужна гибкость в выборе модальностей
  - Качество генерации важнее скорости
