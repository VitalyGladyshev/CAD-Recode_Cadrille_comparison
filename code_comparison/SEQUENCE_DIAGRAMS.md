# Диаграммы последовательности операций генерации CAD-кода

## Оглавление

1. [Последовательность операций в CAD-Recode](#последовательность-операций-в-cad-recode)
2. [Последовательность операций в Cadrille](#последовательность-операций-в-cadrille)
3. [Сравнение процессов генерации](#сравнение-процессов-генерации)

---

## Последовательность операций в CAD-Recode

### Полный цикл генерации

```mermaid
sequenceDiagram
    participant User
    participant Inference as run_inference_batch()
    participant Dataset as STLDataset
    participant Model as CADRecode
    participant Encoder as FourierPointEncoder
    participant BaseModel as Qwen2Model
    participant LMHead as lm_head
    participant Tokenizer as AutoTokenizer

    User->>Inference: Запуск инференса
    Inference->>Dataset: Загрузка STL файла
    Dataset->>Inference: Возврат file_path
    
    Note over Inference: Нормализация mesh
    Inference->>Inference: mesh.apply_translation()\nmesh.apply_scale()
    
    Note over Inference: Генерация точечного облака
    Inference->>Inference: mesh_to_point_cloud(mesh, n_points=256)
    
    Note over Inference: Подготовка входных данных
    Inference->>Inference: input_ids = [pad_token_id] * 256 + [im_start_token]
    Inference->>Inference: attention_mask = [-1] * 256 + [1]
    
    Inference->>Model: generate(input_ids, attention_mask, point_cloud)
    
    loop Для каждого токена генерации
        Model->>Model: forward()
        
        alt Первый проход (past_key_values is None)
            Model->>BaseModel: embed_tokens(input_ids)
            BaseModel-->>Model: inputs_embeds (текстовые)
            
            Model->>Encoder: forward(point_cloud)
            Note over Encoder: Fourier encoding:\ncat([points, sin(freqs*points), cos(freqs*points)])
            Encoder-->>Model: point_embeds (256, hidden_size)
            
            Note over Model: Замена pad токенов на point embeddings
            Model->>Model: inputs_embeds[attention_mask == -1] = point_embeds
            Model->>Model: attention_mask[attention_mask == -1] = 1
            
            Model->>BaseModel: forward(inputs_embeds, attention_mask)
            BaseModel-->>Model: hidden_states
            
            Model->>LMHead: forward(hidden_states)
            LMHead-->>Model: logits
            
            Model->>Model: Выбор следующего токена (sampling)
            Model->>Model: Сохранение в past_key_values
        else Последующие проходы
            Model->>BaseModel: forward(input_ids=new_token, past_key_values)
            BaseModel-->>Model: hidden_states (только для нового токена)
            
            Model->>LMHead: forward(hidden_states)
            LMHead-->>Model: logits
            
            Model->>Model: Выбор следующего токена
        end
    end
    
    Model-->>Inference: generated_ids
    
    Inference->>Tokenizer: extract_code(tokenizer, batch_ids)
    Note over Tokenizer: Удаление служебных токенов:\n<|im_start|>, <|endoftext|>, <|im_end|>
    Tokenizer-->>Inference: py_string (чистый CadQuery код)
    
    Inference->>Inference: Сохранение в файл
    Inference-->>User: Результат генерации
```

### Детальный forward pass CAD-Recode

```mermaid
flowchart TD
    Start([Начало forward]) --> CheckCache{past_key_values<br/>is None?}
    
    CheckCache -->|Да| CreateTextEmbeds[Создание текстовых embeddings<br/>embed_tokens input_ids]
    CheckCache -->|Нет| UseCache[Использование кэша<br/>past_key_values]
    
    CreateTextEmbeds --> EncodePC[Кодирование point cloud<br/>FourierPointEncoder]
    
    EncodePC --> FourierStep[Fourier encoding:<br/>1. Умножение на частоты<br/>2. sin/cos преобразование<br/>3. Объединение с исходными координатами]
    
    FourierStep --> Project[Проекция в hidden_size<br/>nn.Linear 51 -> hidden_size]
    
    Project --> ReplacePads[Замена pad токенов<br/>inputs_embeds где attention_mask == -1<br/>заменяются на point_embeds]
    
    ReplacePads --> UpdateMask[Обновление attention_mask<br/>-1 -> 1]
    
    UpdateMask --> ClearInputIds[Обнуление input_ids<br/>input_ids = None]
    
    ClearInputIds --> CallBase[Вызов базовой модели<br/>Qwen2Model forward]
    
    UseCache --> CallBase
    
    CallBase --> GetLogits[Вычисление логитов<br/>lm_head hidden_states]
    
    GetLogits --> ComputeLoss{labels заданы?}
    
    ComputeLoss -->|Да| CalcLoss[Вычисление CrossEntropyLoss]
    ComputeLoss -->|Нет| NoLoss[loss = None]
    
    CalcLoss --> Return[Возврат CausalLMOutputWithPast]
    NoLoss --> Return
    
    Return --> End([Конец])
```

---

## Последовательность операций в Cadrille

### Полный цикл генерации

```mermaid
sequenceDiagram
    participant User
    participant Inference as run()
    participant Dataset as CadRecodeDataset/Text2CADDataset
    participant Collate as collate()
    participant Processor as Qwen2VLProcessor
    participant Model as Cadrille
    participant VisualEncoder as Visual Encoder
    participant PointEncoder as FourierPointEncoder
    participant BaseModel as Qwen2VLModel
    participant LMHead as lm_head

    User->>Inference: Запуск инференса
    Inference->>Dataset: Загрузка данных
    Dataset->>Inference: Возврат batch элементов
    
    Inference->>Collate: collate(batch, processor, n_points=256, eval=True)
    
    Note over Collate: Формирование сообщений
    Collate->>Collate: Создание chat template<br/>с add_generation_prompt=True
    
    alt Элемент с точечным облаком
        Collate->>Collate: Добавление pad токенов<br/>перед текстом
        Collate->>Collate: is_pc[i] = 1
    else Элемент с изображениями
        Collate->>Collate: Формирование video структуры<br/>is_img[i] = 1
    else Элемент только с текстом
        Collate->>Collate: Стандартная обработка
    end
    
    Collate->>Processor: processor(text, images, videos)
    Processor->>Processor: process_vision_info(messages)
    Processor-->>Collate: pixel_values, pixel_values_videos
    
    Collate->>Processor: Токенизация текста
    Processor-->>Collate: input_ids, attention_mask
    
    Collate-->>Inference: batch с подготовленными данными
    
    Inference->>Model: generate(input_ids, attention_mask,<br/>point_clouds, is_pc, is_img,<br/>pixel_values_videos, video_grid_thw)
    
    loop Для каждого токена генерации
        Model->>Model: forward()
        
        alt Первый проход (past_key_values is None)
            Model->>BaseModel: embed_tokens(input_ids)
            BaseModel-->>Model: inputs_embeds (текстовые)
            
            alt Есть изображения (pixel_values)
                Model->>VisualEncoder: forward(pixel_values, image_grid_thw)
                VisualEncoder-->>Model: image_embeds
                Model->>Model: masked_scatter(image_token_id, image_embeds)
            end
            
            alt Есть видео (pixel_values_videos)
                Model->>VisualEncoder: forward(pixel_values_videos, video_grid_thw)
                VisualEncoder-->>Model: video_embeds
                Model->>Model: masked_scatter(video_token_id, video_embeds)
            end
            
            alt Есть точечные облака (is_pc)
                Model->>PointEncoder: forward(point_clouds.float())
                Note over PointEncoder: Fourier encoding через<br/>FourierEmbedder
                PointEncoder-->>Model: point_embeds
                
                Model->>Model: Вычисление start_idxs<br/>через attention_mask
                Model->>Model: Встраивание через цикл<br/>по батчу
            end
            
            Model->>Model: Вычисление position_ids<br/>с учетом визуальных токенов<br/>get_rope_index()
            
            Model->>BaseModel: forward(inputs_embeds, position_ids, attention_mask)
            BaseModel-->>Model: hidden_states
            
            Model->>LMHead: forward(hidden_states)
            LMHead-->>Model: logits
            
            Model->>Model: Выбор следующего токена
            Model->>Model: Сохранение в past_key_values
        else Последующие проходы
            Model->>BaseModel: forward(input_ids=new_token, past_key_values)
            BaseModel-->>Model: hidden_states (только для нового токена)
            
            Model->>LMHead: forward(hidden_states)
            LMHead-->>Model: logits
            
            Model->>Model: Выбор следующего токена
        end
    end
    
    Model-->>Inference: generated_ids
    
    Inference->>Processor: batch_decode(generated_ids_trimmed,<br/>skip_special_tokens=True)
    Processor-->>Inference: py_strings (CadQuery код)
    
    Inference->>Inference: Сохранение в файлы
    Inference-->>User: Результаты генерации
```

### Детальный forward pass Cadrille

```mermaid
flowchart TD
    Start([Начало forward]) --> CheckInputs{inputs_embeds<br/>заданы?}
    
    CheckInputs -->|Нет| CreateTextEmbeds[Создание текстовых embeddings<br/>embed_tokens input_ids]
    CheckInputs -->|Да| CheckVisual{Есть визуальные<br/>входы?}
    
    CreateTextEmbeds --> CheckVisual
    
    CheckVisual -->|Есть pixel_values| ProcessImages[Обработка изображений<br/>visual encoder]
    CheckVisual -->|Есть pixel_values_videos| ProcessVideos[Обработка видео<br/>visual encoder]
    CheckVisual -->|Есть is_pc| CheckCache{past_key_values<br/>is None?}
    
    ProcessImages --> ValidateImages[Проверка соответствия<br/>image_token_id и features]
    ValidateImages --> EmbedImages[masked_scatter<br/>image_embeds]
    
    ProcessVideos --> ValidateVideos[Проверка соответствия<br/>video_token_id и features]
    ValidateVideos --> EmbedVideos[masked_scatter<br/>video_embeds]
    
    EmbedImages --> CheckCache
    EmbedVideos --> CheckCache
    
    CheckCache -->|Да| EncodePC[Кодирование point cloud<br/>FourierPointEncoder]
    CheckCache -->|Нет| CalcRoPE[Вычисление RoPE<br/>с учетом визуальных токенов]
    
    EncodePC --> FourierStep[Fourier encoding через<br/>FourierEmbedder:<br/>1. Умножение на частоты<br/>2. sin/cos преобразование<br/>3. Объединение]
    
    FourierStep --> Project[Проекция в hidden_size<br/>nn.Linear 51 -> hidden_size]
    
    Project --> CalcStartIdxs[Вычисление start_idxs<br/>attention_mask.shape[1] -<br/>attention_mask.sum axis=1]
    
    CalcStartIdxs --> EmbedPC[Встраивание через цикл<br/>inputs_embeds[i, start_idx:...]<br/>= point_embeds[i]]
    
    EmbedPC --> CalcRoPE
    
    CalcRoPE --> CallBase[Вызов базовой модели<br/>Qwen2VLModel forward<br/>с position_ids]
    
    CallBase --> GetLogits[Вычисление логитов<br/>lm_head hidden_states]
    
    GetLogits --> ComputeLoss{labels<br/>заданы?}
    
    ComputeLoss -->|Да| CalcLoss[Вычисление CrossEntropyLoss]
    ComputeLoss -->|Нет| NoLoss[loss = None]
    
    CalcLoss --> Return[Возврат Qwen2VLCausalLMOutputWithPast]
    NoLoss --> Return
    
    Return --> End([Конец])
```

---

## Сравнение процессов генерации

### Подготовка входных данных

```mermaid
flowchart LR
    subgraph CADRecode[CAD-Recode]
        A1[STL файл] --> A2[Нормализация mesh]
        A2 --> A3[mesh_to_point_cloud<br/>256 точек]
        A3 --> A4[Создание input_ids<br/>pad_token_id * 256 + im_start]
        A4 --> A5[Создание attention_mask<br/>-1 * 256 + 1]
        A5 --> A6[Готовые данные]
    end
    
    subgraph Cadrille[Cadrille]
        B1[Данные из датасета] --> B2{Тип данных?}
        B2 -->|PC| B3[Добавление pad токенов<br/>в collate]
        B2 -->|IMG| B4[Рендеринг mesh<br/>4 ракурса]
        B2 -->|TEXT| B5[Только текст]
        B3 --> B6[Формирование chat template]
        B4 --> B6
        B5 --> B6
        B6 --> B7[process_vision_info<br/>для изображений/видео]
        B7 --> B8[Токенизация через processor]
        B8 --> B9[Готовые данные]
    end
```

### Интеграция point embeddings

```mermaid
flowchart TD
    subgraph CADRecodeMethod[CAD-Recode метод]
        C1[inputs_embeds = embed_tokens input_ids] --> C2[point_embeds = FourierPointEncoder]
        C2 --> C3[Векторная операция:<br/>inputs_embeds[attention_mask == -1]<br/>= point_embeds.reshape]
        C3 --> C4[attention_mask[-1] = 1]
        C4 --> C5[input_ids = None]
    end
    
    subgraph CadrilleMethod[Cadrille метод]
        D1[inputs_embeds = embed_tokens input_ids] --> D2[Обработка изображений/видео<br/>если есть]
        D2 --> D3[point_embeds = FourierPointEncoder]
        D3 --> D4[Вычисление start_idxs:<br/>attention_mask.shape[1] -<br/>attention_mask.sum axis=1]
        D4 --> D5[Цикл по батчу:<br/>inputs_embeds[i, start_idx:...]<br/>= point_embeds[i]]
    end
```

### Обработка кэша при генерации

```mermaid
sequenceDiagram
    participant Model
    participant Cache as past_key_values
    participant Transformer
    
    Note over Model: Первый токен генерации
    Model->>Model: past_key_values = None
    Model->>Transformer: forward(inputs_embeds,<br/>point_embeds встроены)
    Transformer->>Cache: Сохранение ключей/значений
    Transformer-->>Model: hidden_states, past_key_values
    
    Note over Model: Второй токен генерации
    Model->>Model: past_key_values != None
    Model->>Transformer: forward(input_ids=new_token,<br/>past_key_values)
    Note over Transformer: point_embeds НЕ встраиваются<br/>используется кэш
    Transformer->>Cache: Обновление кэша
    Transformer-->>Model: hidden_states, past_key_values
    
    Note over Model: Последующие токены...
    Model->>Transformer: forward(input_ids=new_token,<br/>past_key_values)
    Transformer-->>Model: hidden_states, past_key_values
```

---

## Ключевые различия в последовательностях

### 1. Подготовка данных

| Этап | CAD-Recode | Cadrille |
|------|------------|----------|
| Нормализация | В `run_inference_batch()` | В датасете (`CadRecodeDataset`) |
| Генерация PC | В `run_inference_batch()` | В датасете (`get_point_cloud()`) |
| Создание токенов | Прямое создание списков | Через `processor()` и `collate()` |
| Chat template | Нет | Да (`apply_chat_template()`) |
| Обработка визуальных данных | Нет | Да (`process_vision_info()`) |

### 2. Forward pass

| Этап | CAD-Recode | Cadrille |
|------|------------|----------|
| Текстовые embeddings | `embed_tokens(input_ids)` | `embed_tokens(input_ids)` |
| Визуальные embeddings | Нет | `visual(pixel_values/videos)` |
| Point embeddings | `FourierPointEncoder` | `FourierPointEncoder` |
| Интеграция PC | Векторная операция `[mask == -1]` | Цикл по батчу с `start_idxs` |
| RoPE | Стандартная | С учетом визуальных токенов |
| Базовый трансформер | `Qwen2Model` | `Qwen2VLModel` |

### 3. Декодирование

| Этап | CAD-Recode | Cadrille |
|------|------------|----------|
| Метод | `extract_code()` (ручная очистка) | `processor.batch_decode()` |
| Удаление токенов | Ручное удаление `<|im_start|>`, `<|endoftext|>` | Автоматическое через `skip_special_tokens=True` |
| Обработка батча | По одному элементу | Векторизованная обработка |

---

## Оптимизации генерации

### Использование кэша (past_key_values)

**Обе модели:**

1. **Первый проход (prefill):**
   - Point embeddings встраиваются в последовательность
   - Все входные токены обрабатываются
   - Ключи и значения сохраняются в `past_key_values`

2. **Последующие проходы (decode):**
   - Point embeddings **не** встраиваются повторно
   - Обрабатывается только новый токен
   - Используются сохраненные ключи/значения из кэша
   - Это значительно ускоряет генерацию

### Различия в оптимизациях

**CAD-Recode:**
- Простая структура кэша
- Только текстовые токены в кэше
- Минимальные вычисления на каждом шаге

**Cadrille:**
- Более сложная структура кэша
- Визуальные токены также кэшируются
- RoPE вычисляется с учетом визуальных токенов
- Больше вычислений на первом проходе, но эффективная генерация
