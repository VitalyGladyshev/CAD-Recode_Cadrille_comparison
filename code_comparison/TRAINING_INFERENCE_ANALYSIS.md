# Анализ подходов к обучению и инференсу

## Оглавление

1. [Подготовка данных для обучения](#подготовка-данных-для-обучения)
2. [Процесс инференса](#процесс-инференса)
3. [Оптимизации для обучения и инференса](#оптимизации-для-обучения-и-инференса)
4. [Сравнительный анализ](#сравнительный-анализ)

---

## Подготовка данных для обучения

### Cadrille: Формирование labels

**Файлы:** `cadrille.py`: функция `collate()` (строки 40-226), `find_assistant_content_sublist_indexes()` (строки 229-276)

**Процесс формирования labels:**

1. **Создание сообщений с ответами** (строки 107-142):
   ```python
   # Режим обучения: создаем полные диалоги с ответами ассистента
   for i, m in enumerate(batch):
       if 'video' in m.keys():
           # Элемент с изображениями/видео
           message = [{
               'role': 'user',
               'content': [
                   {'type': 'video', 'video': m['video'], 'fps': 1.0},
                   {'type': 'text', 'text': m['description']}
               ]
           }, {
               'role': 'assistant',
               'content': [
                   {'type': 'text', 'text': m['answer']}
               ]
           }]
       else:
           # Элемент с точечным облаком или текстом
           message = [{
               'role': 'user',
               'content': [
                   {'type': 'text', 'text': m['description']}
               ]
           }, {
               'role': 'assistant',
               'content': [
                   {'type': 'text', 'text': m['answer']}
               ]
           }]
   ```

2. **Применение chat template** (строка 145):
   ```python
   texts = [
       processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
       for msg in messages
   ]
   ```
   - `add_generation_prompt=False`: Не добавляется промпт генерации, так как ответ уже присутствует

3. **Токенизация** (строки 185-190):
   ```python
   inputs = processor(
       text=texts,
       images=image_inputs,
       videos=video_inputs,
       padding=True,
       return_tensors='pt'
   )
   ```

4. **Формирование labels** (строки 211-223):
   ```python
   input_ids_lists = inputs['input_ids'].tolist()
   labels_list = []
   for ids_list in input_ids_lists:
       label_ids = [-100] * len(ids_list)  # -100 игнорируется в loss
       for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
           # Labels только для содержимого ассистента
           label_ids[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1] = \
               ids_list[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1]
       labels_list.append(label_ids)
   labels_ids = torch.tensor(labels_list, dtype=torch.int64)
   inputs['labels'] = labels_ids
   ```

**Функция `find_assistant_content_sublist_indexes()`** (строки 229-276):

```python
def find_assistant_content_sublist_indexes(l):
    """
    Находит индексы начала и конца содержимого ассистента.
    
    Ищет пары токенов (151644, 77091) как маркеры начала ответа
    и следующий токен 151645 как маркер конца ответа.
    """
    start_indexes = []
    end_indexes = []
    
    for i in range(len(l) - 1):
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break
    
    return list(zip(start_indexes, end_indexes))
```

**Особенности:**
- Labels создаются только для содержимого ассистента (CadQuery код)
- Промпт пользователя и системные токены игнорируются (`-100`)
- Используются специальные токены Qwen2VL для маркировки ответов
- Поддерживает множественные ответы в одной последовательности

### CAD-Recode: Формирование labels

**Примечание:** В текущем коде инференса (`test_cad_recode.py`) нет явного формирования labels, так как это код для тестирования. Однако, для обучения используется аналогичный подход:

1. **Подготовка данных:**
   - Загрузка CadQuery кода из файла (`item['py_path']`)
   - Создание последовательности: pad токены + промпт + код

2. **Формирование labels:**
   - Аналогично Cadrille: labels только для целевого кода
   - Промпт и pad токены игнорируются
   - Используется стандартный подход с `-100` для игнорируемых токенов

**Различия:**
- CAD-Recode не использует chat template
- Нет специальных токенов для маркировки ответов (151644, 77091, 151645)
- Более простая структура labels

### Chat template в Cadrille

**Режим обучения** (`add_generation_prompt=False`):
```
<|im_start|>user
{description}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>
```

**Режим инференса** (`add_generation_prompt=True`):
```
<|im_start|>user
{description}
<|im_end|>
<|im_start|>assistant
```

**Особенности:**
- Автоматическое форматирование диалога
- Обработка специальных токенов
- Поддержка мультимодальных входов в template

### Обработка батчей

**Cadrille:**
- Поддержка смешанных батчей (PC + IMG + TEXT)
- Маски `is_pc` и `is_img` определяют тип каждого элемента
- Обработка через условные проверки в `collate()` и `forward()`

**CAD-Recode:**
- Только PC батчи
- Все элементы батча содержат точечные облака
- Нет необходимости в масках для типов модальностей

---

## Процесс инференса

### Cadrille: Подготовка входных данных

**Файлы:** `test.py`: функция `run()` (строки 50-238), `cadrille.py`: функция `collate()` (строки 40-226)

**Процесс:**

1. **Загрузка датасета** (строки 121-138):
   ```python
   if mode == 'text':
       dataset = Text2CADDataset(...)
       batch_size = 32
   else:  # mode in ('pc', 'img')
       dataset = CadRecodeDataset(...)
       batch_size = 8
   ```

2. **Создание DataLoader** (строки 152-156):
   ```python
   dataloader = DataLoader(
       dataset=concat_dataset,
       batch_size=batch_size,
       num_workers=20,
       collate_fn=partial(collate, processor=processor, n_points=256, eval=True)
   )
   ```

3. **Подготовка в collate()** (`eval=True`):
   - Формирование сообщений только с запросами пользователя (без ответов)
   - Применение chat template с `add_generation_prompt=True`
   - Добавление pad токенов для PC элементов
   - Обработка визуальных данных через `process_vision_info()`

4. **Генерация** (строки 161-169):
   ```python
   generated_ids = model.generate(
       input_ids=batch['input_ids'].to(model.device),
       attention_mask=batch['attention_mask'].to(model.device),
       point_clouds=batch['point_clouds'].to(model.device),
       is_pc=batch['is_pc'].to(model.device),
       is_img=batch['is_img'].to(model.device),
       pixel_values_videos=batch['pixel_values_videos'].to(model.device),
       video_grid_thw=batch['video_grid_thw'].to(model.device),
       max_new_tokens=768
   )
   ```

5. **Декодирование** (строки 172-179):
   ```python
   generated_ids_trimmed = [
       out_ids[len(in_ids):] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
   ]
   py_strings = processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
   )
   ```

**Особенности:**
- Использование `collate()` для подготовки батчей
- Автоматическая обработка различных типов входов
- Векторизованное декодирование через `processor.batch_decode()`

### CAD-Recode: Подготовка входных данных

**Файлы:** `test_cad_recode.py`: функции `run_inference_batch()` (строки 527-632), `run_cad_recode_inference()` (строки 635-743)

**Процесс:**

1. **Загрузка датасета** (строка 723):
   ```python
   dataset = STLDataset(data_path, split, max_samples)
   dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=20)
   ```

2. **Подготовка для каждого элемента** (`run_inference_batch()`, строки 595-608):
   ```python
   # Загрузка и нормализация mesh
   gt_mesh = trimesh.load_mesh(file_path)
   gt_mesh.apply_translation(-(gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2.0)
   gt_mesh.apply_scale(2.0 / max(gt_mesh.extents))
   
   # Генерация точечного облака
   point_cloud = mesh_to_point_cloud(gt_mesh, n_points=256)
   
   # Подготовка входных данных
   input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
   attention_mask = [-1] * len(point_cloud) + [1]
   ```

3. **Генерация** (строки 612-617):
   ```python
   batch_ids = model.generate(
       input_ids=torch.tensor(input_ids).unsqueeze(0).to(device),
       attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(device),
       point_cloud=torch.tensor(point_cloud.astype(np.float32)).unsqueeze(0).to(device),
       max_new_tokens=768,
       pad_token_id=tokenizer.pad_token_id
   )
   ```

4. **Декодирование** (строка 620):
   ```python
   py_string = extract_code(tokenizer, batch_ids)
   ```

**Функция `extract_code()`** (строки 383-454):

```python
def extract_code(tokenizer, batch_ids):
    """
    Извлекает CadQuery код из токенизированного вывода модели.
    
    Удаляет все служебные токены и извлекает чистый Python код.
    """
    # Декодируем с пропуском специальных токенов
    py_string = tokenizer.batch_decode(batch_ids, skip_special_tokens=False)[0]
    
    # Удаляем все служебные токены
    if '<|im_start|>' in py_string:
        start_idx = py_string.find('<|im_start|>')
        if start_idx != -1:
            py_string = py_string[start_idx + len('<|im_start|>'):]
    
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
        import_idx = py_string.find('import')
        if import_idx != -1:
            py_string = py_string[import_idx:]
    
    return py_string.strip()
```

**Особенности:**
- Прямая подготовка входных данных без `collate()`
- Ручная обработка каждого элемента батча
- Ручная очистка токенов в `extract_code()`

### Сравнение декодирования

| Аспект | Cadrille | CAD-Recode |
|--------|----------|------------|
| Метод | `processor.batch_decode()` | `extract_code()` (ручная обработка) |
| Обработка батча | Векторизованная | По одному элементу |
| Удаление токенов | Автоматическое (`skip_special_tokens=True`) | Ручное удаление через `replace()` и `find()` |
| Очистка пробелов | `clean_up_tokenization_spaces=False` | `strip()` в конце |
| Поиск начала кода | Не требуется (chat template) | Поиск `import` или `import cadquery` |

---

## Оптимизации для обучения и инференса

### Использование памяти

**Обе модели:**

1. **Типы данных:**
   - Модели работают в `bfloat16` для экономии памяти GPU
   - `FourierPointEncoder` создается в `float32` для стабильности вычислений
   - Автоматическая конвертация в нужный dtype при использовании

2. **Flash Attention 2:**
   ```python
   # Cadrille
   model = Cadrille.from_pretrained(
       ...,
       attn_implementation='flash_attention_2' if device == 'cuda' else None
   )
   
   # CAD-Recode
   model = CADRecode.from_pretrained(
       ...,
       attn_implementation='flash_attention_2' if device == 'cuda' else None
   )
   ```
   - Ускоряет вычисления attention
   - Экономит память за счет переиспользования промежуточных результатов

3. **Очистка памяти GPU:**
   - Использование `torch.cuda.empty_cache()` между этапами
   - `gc.collect()` для освобождения памяти Python

### Параллелизация

**DataLoader:**
- Cadrille: `num_workers=20` для всех режимов
- CAD-Recode: `num_workers=20`
- Использование всех доступных ядер CPU для загрузки данных

**Batch sizes:**
- Cadrille:
  - `batch_size=32` для текстового режима
  - `batch_size=8` для PC и IMG режимов
- CAD-Recode:
  - `batch_size=8` для всех случаев

**Обоснование:**
- Меньший batch size для PC/IMG из-за больших размеров данных (point clouds, изображения)
- Больший batch size для текста из-за меньшего размера данных

### Кэширование при генерации

**Обе модели:**

1. **past_key_values:**
   - Кэширование ключей и значений attention для ускорения генерации
   - Point embeddings встраиваются только на первом проходе
   - На последующих итерациях используются только новые токены

2. **Условие встраивания PC:**
   ```python
   if past_key_values is None or past_key_values.get_seq_length() == 0:
       # Встраивание point embeddings
   ```
   - Проверка на первый проход
   - Избежание повторных вычислений

3. **Инкрементальная генерация:**
   - Обработка только нового токена на каждой итерации
   - Использование сохраненных ключей/значений из кэша
   - Значительное ускорение генерации длинных последовательностей

### Оптимизации специфичные для Cadrille

1. **Обработка визуальных данных:**
   - Визуальные токены также кэшируются
   - RoPE вычисляется один раз с учетом визуальных токенов
   - `rope_deltas` сохраняются для последующих итераций

2. **Смешанные батчи:**
   - Эффективная обработка различных типов входов в одном батче
   - Условные проверки только при необходимости (`if is_pc.sum() > 0`)

### Оптимизации специфичные для CAD-Recode

1. **Простая структура:**
   - Минимальные вычисления на каждом шаге
   - Нет обработки визуальных данных
   - Быстрая генерация за счет простоты

2. **Векторные операции:**
   - Встраивание PC через векторную операцию с маской
   - Эффективная обработка батчей

---

## Сравнительный анализ

### Подготовка данных для обучения

| Аспект | Cadrille | CAD-Recode |
|--------|----------|------------|
| Chat template | Да (`apply_chat_template()`) | Нет |
| Формирование labels | Через `find_assistant_content_sublist_indexes()` | Прямое формирование |
| Специальные токены | 151644, 77091, 151645 | Стандартные |
| Поддержка модальностей | PC + IMG + TEXT | Только PC |
| Обработка батчей | Смешанные батчи | Только PC батчи |

### Процесс инференса

| Аспект | Cadrille | CAD-Recode |
|--------|----------|------------|
| Подготовка данных | `collate()` функция | Прямая подготовка |
| Chat template | Да (`add_generation_prompt=True`) | Нет |
| Параметры генерации | `point_clouds`, `is_pc`, `is_img`, `pixel_values_videos` | `point_cloud`, `attention_mask` |
| Декодирование | `processor.batch_decode()` | `extract_code()` (ручная) |
| Обработка батча | Векторизованная | По одному элементу |

### Оптимизации

| Аспект | Cadrille | CAD-Recode |
|--------|----------|------------|
| Тип данных модели | `bfloat16` | `bfloat16` |
| Fourier encoder dtype | `float32` → `bfloat16` | `float32` → `dtype модели` |
| Flash Attention | Да | Да |
| num_workers | 20 | 20 |
| Batch size (PC) | 8 | 8 |
| Batch size (TEXT) | 32 | N/A |
| Кэширование | past_key_values + rope_deltas | past_key_values |

### Производительность

**Cadrille:**
- Больше вычислений на первом проходе (обработка визуальных данных)
- Эффективная генерация благодаря кэшированию
- Поддержка смешанных батчей увеличивает гибкость

**CAD-Recode:**
- Меньше вычислений на первом проходе
- Простая структура обеспечивает быструю генерацию
- Векторные операции для эффективной обработки батчей

---

## Выводы

### Подходы к обучению

1. **Cadrille:**
   - Использует chat template для структурирования диалогов
   - Labels формируются только для содержимого ассистента
   - Поддержка обучения на смешанных модальностях

2. **CAD-Recode:**
   - Более простая структура без chat template
   - Прямое формирование labels
   - Фокус только на точечных облаках

### Подходы к инференсу

1. **Cadrille:**
   - Использует `collate()` для подготовки батчей
   - Автоматическая обработка различных типов входов
   - Векторизованное декодирование

2. **CAD-Recode:**
   - Прямая подготовка входных данных
   - Ручная обработка каждого элемента
   - Ручная очистка токенов

### Рекомендации по оптимизации

1. **Для обучения:**
   - Использовать `bfloat16` для экономии памяти
   - Flash Attention 2 для ускорения
   - Оптимизировать batch size в зависимости от доступной памяти

2. **Для инференса:**
   - Использовать кэширование (`past_key_values`)
   - Оптимизировать количество воркеров для DataLoader
   - Использовать векторизованные операции где возможно
