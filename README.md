# Конвейер тестирования и оценки моделей CAD-Recode и Cadrille

## Описание проекта

Проект представляет собой полный конвейер для тестирования и оценки моделей генерации CAD-кода: **CAD-Recode** и **Cadrille**. Решение обеспечивает сопоставимые результаты и визуализацию метрик для объективного сравнения производительности моделей на различных датасетах.

### Основные возможности

- **Полный цикл тестирования**: от загрузки данных до визуализации результатов
- **Сопоставимая оценка**: единые метрики и процедуры для обеих моделей
- **Множественные метрики**: Chamfer Distance, IoU, Invalidity Ratio
- **Визуализация результатов**: графики сравнения и радарные диаграммы
- **Оптимизация производительности**: поддержка GPU, параллельная обработка
- **Надежность**: обработка ошибок, таймауты, валидация данных

## Структура проекта

```
cad_unified/
├── results/                   # Результаты сравнения
├── test.py                    # Тестирование модели Cadrille
├── test_cad_recode.py         # Тестирование модели CAD-Recode
├── evaluate.py                # Оценка метрик для обеих моделей
├── compare_results.py         # Сравнение и визуализация результатов
├── pipeline.py                # Полный конвейер тестирования и оценки
├── dataset.py                 # Классы датасетов для загрузки данных
├── cadrille.py                # Модель Cadrille и вспомогательные функции
├── Dockerfile                 # Docker-образ для развертывания
├── .gitignore                 # Игнорируемые файлы для Git
└── README.md                  # Документация проекта
```

### Основные модули

- **`test.py`**: Запуск инференса модели Cadrille на датасетах. Поддерживает режимы работы с текстом, точечными облаками и изображениями.
- **`test_cad_recode.py`**: Запуск инференса модели CAD-Recode. Генерирует CadQuery коды из точечных облаков 3D моделей.
- **`evaluate.py`**: Вычисление метрик качества (Chamfer Distance, IoU) для сгенерированных моделей. Поддерживает параллельную обработку.
- **`compare_results.py`**: Сравнение результатов оценки двух моделей с созданием визуализаций и текстовых отчетов.
- **`pipeline.py`**: Оркестрация полного цикла: тестирование обеих моделей → оценка → сравнение результатов.
- **`dataset.py`**: Классы датасетов для загрузки данных (CadRecodeDataset, Text2CADDataset) и утилиты для работы с mesh.
- **`cadrille.py`**: Реализация модели Cadrille, функции коллации данных и обработки мультимодальных входов.

## Требования к системе

### Аппаратные требования

- **GPU**: NVIDIA GPU с поддержкой CUDA (рекомендуется RTX 3090 Ti или лучше)
- **Память**: Минимум 16 GB RAM, рекомендуется 32 GB+
- **Дисковое пространство**: Минимум 50 GB для данных и моделей

### Программное обеспечение

- **Python**: 3.11+
- **CUDA**: 12.4+ (для GPU)
- **Docker**: Опционально, для использования готового окружения

### Основные зависимости

Проект использует следующие ключевые библиотеки:

- `torch` >= 2.5.1 - PyTorch для работы с моделями
- `transformers` >= 4.50.3 - Hugging Face Transformers для моделей
- `trimesh` >= 4.5.3 - Работа с 3D моделями
- `open3d` - Рендеринг и обработка 3D данных
- `pytorch3d` - 3D операции для PyTorch
- `cadquery` - Генерация и выполнение CadQuery кода
- `matplotlib` >= 3.10.0 - Визуализация результатов
- `numpy` >= 2.2.0 - Численные вычисления
- `scipy` >= 1.14.1 - Научные вычисления

Полный список зависимостей указан в `Dockerfile`.

## Установка

### Вариант 1: Использование Docker (рекомендуется)

1. Соберите Docker-образ:
```bash
docker build -t cad-unified .
```

2. Запустите контейнер:
```bash
docker run -it --gpus all \
  -v /path/to/data:/workspace/data \
  -v /path/to/models:/workspace/models \
  -v /path/to/results:/workspace/results \
  -p 8888:8888 \
  cad-unified
```

### Вариант 2: Локальная установка

1. Установите зависимости из Dockerfile или создайте `requirements.txt`:
```bash
pip install torch transformers trimesh open3d pytorch3d cadquery matplotlib numpy scipy
```

2. Установите дополнительные зависимости:
```bash
pip install flash-attn --no-build-isolation
pip install git+https://github.com/facebookresearch/pytorch3d@06a76ef8ddd00b6c889768dfc990ae8cb07c6f2f
pip install git+https://github.com/CadQuery/cadquery.git@e99a15df3cf6a88b69101c405326305b5db8ed94
pip install cadquery-ocp==7.7.2
```

## Быстрый старт

### 1. Проверка окружения

Убедитесь, что все зависимости установлены и GPU доступен:

```bash
# Проверка GPU
nvidia-smi

# Проверка установленных пакетов
python -c "import torch; print(f'PyTorch версия: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}')"
```

### 2. Подготовка данных

Убедитесь, что данные находятся в правильной структуре:

```
/workspace/data/
├── deepcad_test_mesh/
│   ├── 00000001.stl
│   ├── 00000002.stl
│   └── ...
└── fusion360_test_mesh/
    ├── 00000001.stl
    └── ...
```

### 3. Подготовка моделей

Поместите модели в соответствующие директории:

```
/workspace/models/
├── cad-recode-v1.5/          # Модель CAD-Recode
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── cadrille-rl/              # Модель Cadrille (Hugging Face)
```

## Использование

### Запуск полного конвейера

Самый простой способ - использовать `pipeline.py`, который автоматически запускает все этапы:

#### Для DeepCAD датасета

```bash
python pipeline.py \
  --data-path /workspace/data \
  --models-dir /workspace/models \
  --results-dir /workspace/results \
  --dataset deepcad_test_mesh \
  --n-samples 1 \
  --max-samples 50 \
  --device cuda
```

#### Для Fusion360 датасета

```bash
python pipeline.py \
  --data-path /workspace/data \
  --models-dir /workspace/models \
  --results-dir /workspace/results \
  --dataset fusion360_test_mesh \
  --n-samples 1 \
  --max-samples 50 \
  --device cuda
```

**Параметры `pipeline.py`:**
- `--data-path`: Путь к директории с данными
- `--models-dir`: Путь к директории с моделями
- `--results-dir`: Путь для сохранения результатов
- `--dataset`: Название датасета (`deepcad_test_mesh` или `fusion360_test_mesh`)
- `--n-samples`: Количество генераций на объект (по умолчанию: 1)
- `--max-samples`: Максимальное количество объектов для обработки (по умолчанию: 50)
- `--device`: Устройство для вычислений (`cuda` или `cpu`)

### Запуск по отдельным этапам

Если нужно запустить только определенные этапы:

#### 1. Тестирование CAD-Recode

```bash
python test_cad_recode.py \
  --data-path /workspace/data \
  --split deepcad_test_mesh \
  --model-path /workspace/models/cad-recode-v1.5 \
  --output-dir /workspace/results/cad_recode/deepcad_test_mesh \
  --device cuda \
  --n-samples 1 \
  --max-samples 50
```

**Параметры `test_cad_recode.py`:**
- `--data-path`: Путь к директории с данными
- `--split`: Название датасета для тестирования
- `--model-path`: Путь к модели CAD-Recode
- `--output-dir`: Директория для сохранения результатов
- `--device`: Устройство для вычислений (`cuda`/`cpu`)
- `--n-samples`: Количество генераций на объект
- `--max-samples`: Максимальное количество объектов

#### 2. Оценка CAD-Recode

```bash
python evaluate.py \
  --gt-mesh-path /workspace/data/deepcad_test_mesh \
  --pred-py-path /workspace/results/cad_recode/deepcad_test_mesh \
  --pred-eval-path /workspace/results/cad_recode_eval/deepcad_test_mesh \
  --n-points 8192 \
  --num-workers 20
```

**Параметры `evaluate.py`:**
- `--gt-mesh-path`: Путь к ground truth mesh файлам (STL)
- `--pred-py-path`: Путь к предсказанным CadQuery кодам (.py файлы)
- `--pred-eval-path`: Путь для сохранения результатов оценки
- `--n-points`: Количество точек для Chamfer Distance (по умолчанию: 8192)
- `--num-workers`: Количество процессов для параллельной обработки (по умолчанию: 20)
- `--check-models`: Опционально: проверить несколько файлов на корректность
- `--debug`: Опционально: включить режим отладки

#### 3. Тестирование Cadrille

```bash
python test.py \
  --data-path /workspace/data \
  --split deepcad_test_mesh \
  --mode pc \
  --checkpoint-path /workspace/models/cadrille-rl \
  --py-path /workspace/results/cadrille/deepcad_test_mesh \
  --device cuda \
  --max-samples 50
```

**Параметры `test.py`:**
- `--data-path`: Путь к корневой директории с данными
- `--split`: Название датасета для тестирования
- `--mode`: Режим работы модели (`text`, `pc`, `img`)
- `--checkpoint-path`: Путь к чекпоинту модели Cadrille (Hugging Face)
- `--py-path`: Директория для сохранения сгенерированных CadQuery файлов
- `--device`: Устройство для вычислений (`cuda`/`cpu`)
- `--max-samples`: Максимальное количество объектов для обработки

#### 4. Оценка Cadrille

```bash
python evaluate.py \
  --gt-mesh-path /workspace/data/deepcad_test_mesh \
  --pred-py-path /workspace/results/cadrille/deepcad_test_mesh \
  --pred-eval-path /workspace/results/cadrille_eval/deepcad_test_mesh \
  --n-points 8192 \
  --num-workers 20
```

#### 5. Сравнение результатов

```bash
python compare_results.py \
  --cad-recode-results /workspace/results/cad_recode_eval/deepcad_test_mesh/summary.txt \
  --cadrille-results /workspace/results/cadrille_eval/deepcad_test_mesh/summary.txt \
  --output-dir /workspace/results/comparison/deepcad_test_mesh \
  --dataset deepcad_test_mesh
```

**Параметры `compare_results.py`:**
- `--cad-recode-results`: Путь к `summary.txt` для CAD-Recode
- `--cadrille-results`: Путь к `summary.txt` для Cadrille
- `--output-dir`: Директория для сохранения результатов сравнения
- `--dataset`: Название датасета для заголовков

## Метрики оценки

Проект использует следующие метрики для оценки качества генерации:

### Chamfer Distance (CD)

Метрика для оценки качества реконструкции 3D моделей. Измеряет среднее квадратичное расстояние от каждой точки одной модели до ближайшей точки другой модели. **Меньшее значение означает лучшее соответствие.**

- **Единицы измерения**: миллиметры (мм)
- **Типичные значения**: 0.1 - 10.0 мм для хороших результатов
- **Вычисление**: Двунаправленная метрика (от GT к Pred и от Pred к GT)

### Intersection over Union (IoU)

Метрика для оценки перекрытия между двумя 3D моделями. Вычисляется как отношение объема пересечения к объему объединения моделей. **Значения находятся в диапазоне [0, 1], где 1 означает полное совпадение.**

- **Типичные значения**: 0.5 - 0.9 для хороших результатов
- **Методы вычисления**:
  - **Point-based**: Плотное точечное облако внутри bounding box (по умолчанию)
  - **Bounding box**: Упрощенная версия на основе пересечения bounding boxes

### Invalidity Ratio

Доля невалидных моделей среди всех сгенерированных. Модель считается невалидной, если:
- CadQuery код не может быть выполнен
- Результирующая 3D модель пустая или некорректная
- Не удалось вычислить метрики

- **Единицы измерения**: проценты (%)
- **Типичные значения**: 0% - 20% для хороших результатов
- **Формула**: `(количество невалидных моделей / общее количество моделей) * 100%`

## Структура результатов

После выполнения конвейера создается следующая структура директорий:

```
/workspace/results/
├── cad_recode/
│   └── deepcad_test_mesh/
│       ├── 00000001+0.py
│       ├── 00000002+0.py
│       └── ...
├── cad_recode_eval/
│   └── deepcad_test_mesh/
│       ├── meshes/              # Конвертированные STL файлы
│       ├── breps/              # STEP файлы
│       ├── summary.txt         # Текстовый отчет с метриками
│       ├── detailed_results.json
│       └── metrics_distribution.png
├── cadrille/
│   └── deepcad_test_mesh/
│       ├── 00000001+0.py
│       └── ...
├── cadrille_eval/
│   └── deepcad_test_mesh/
│       ├── meshes/
│       ├── breps/
│       ├── summary.txt
│       └── ...
└── comparison/
    └── deepcad_test_mesh/
        ├── chamfer_distance_comparison.png
        ├── iou_comparison.png
        ├── invalidity_ratio_comparison.png
        ├── radar_comparison.png
        ├── comparison_results.json
        └── comparison_report.txt
```

## Результаты

После выполнения конвейера в директории `/workspace/results/comparison/{dataset_name}` созданы:

1. **`chamfer_distance_comparison.png`** - Сравнение Chamfer Distance между моделями
2. **`iou_comparison.png`** - Сравнение IoU между моделями
3. **`invalidity_ratio_comparison.png`** - Сравнение Invalidity Ratio между моделями
4. **`radar_comparison.png`** - Радарная диаграмма комплексного сравнения всех метрик
5. **`comparison_results.json`** - Структурированные данные для анализа
6. **`comparison_report.txt`** - Подробный текстовый отчет с выводами

Эти результаты позволят объективно сравнить производительность моделей CAD-Recode и Cadrille на выбранном датасете и сделать выводы об их относительной эффективности.

## Устранение неполадок

### Проблемы с памятью GPU

Если возникают ошибки нехватки памяти:

1. Уменьшите размер батча в соответствующих скриптах
2. Используйте `--max-samples` для ограничения количества обрабатываемых объектов
3. Убедитесь, что используется `bfloat16` для моделей

### Проблемы с конвертацией CadQuery

Если многие модели не конвертируются:

1. Проверьте логи в `evaluate.py` для детальной информации об ошибках
2. Используйте `--check-models` для предварительной проверки файлов
3. Увеличьте таймаут в `evaluate.py` (по умолчанию 5 секунд)

### Проблемы с параллельной обработкой

Если возникают проблемы с multiprocessing:

1. Уменьшите `--num-workers` до меньшего значения
2. Используйте `--debug` для последовательной обработки
3. Проверьте, что система имеет достаточно ресурсов

## Заключение

Предложенный конвейер обеспечивает полный цикл тестирования и оценки моделей CAD-Recode и Cadrille с получением сопоставимых результатов. Решение оптимизировано для работы на GPU RTX 3090 и обеспечивает надежную обработку данных с детальной визуализацией результатов для объективного сравнения производительности моделей.
