#!/usr/bin/env python3
"""
Модуль для работы с датасетами 3D моделей для обучения и тестирования моделей CAD.

Предоставляет классы и функции для загрузки, обработки и преобразования 3D моделей
в различные форматы: точечные облака, изображения и текстовые описания.

Основные компоненты:
    - mesh_to_point_cloud(): конвертация mesh в точечное облако
    - mesh_to_image(): рендеринг mesh в изображение
    - CadRecodeDataset: датасет для работы с точечными облаками и изображениями
    - Text2CADDataset: датасет для работы с текстовыми описаниями

Используемые библиотеки:
    - trimesh: для работы с 3D моделями
    - open3d: для рендеринга mesh в изображения
    - pytorch3d: для farthest point sampling
    - PIL: для обработки изображений
    - torch: для работы с датасетами PyTorch

Пример использования:
    >>> from dataset import CadRecodeDataset
    >>> dataset = CadRecodeDataset(
    ...     root_dir='/workspace/data',
    ...     split='deepcad_test_mesh',
    ...     n_points=256,
    ...     normalize_std_pc=100,
    ...     noise_scale_pc=None,
    ...     img_size=128,
    ...     normalize_std_img=200,
    ...     noise_scale_img=-1,
    ...     num_imgs=4,
    ...     mode='pc'
    ... )
    >>> item = dataset[0]
"""
import os
import pickle
import open3d
import trimesh
import skimage
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points


def mesh_to_point_cloud(mesh: 'trimesh.Trimesh', n_points: int, n_pre_points: int = 8192) -> np.ndarray:
    """
    Конвертирует 3D mesh в точечное облако с использованием farthest point sampling.
    
    Использует двухэтапный процесс:
    1. Предварительное сэмплирование большого количества точек с поверхности mesh
    2. Farthest point sampling для выбора равномерно распределенных точек
    
    Farthest point sampling обеспечивает более равномерное распределение точек
    по сравнению с простым случайным сэмплированием, что важно для качества
    представления геометрии модели.
    
    Args:
        mesh (trimesh.Trimesh): Исходный 3D mesh для конвертации.
            Должен быть валидной моделью с вершинами и гранями.
        n_points (int): Количество точек в итоговом точечном облаке.
            Типичные значения: 256, 512, 1024.
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
        Функция использует trimesh для сэмплирования поверхности и pytorch3d
        для farthest point sampling. Результат всегда содержит ровно n_points точек.
    
    Example:
        >>> import trimesh
        >>> mesh = trimesh.load('model.stl')
        >>> point_cloud = mesh_to_point_cloud(mesh, n_points=256)
        >>> print(f"Точечное облако: {point_cloud.shape}")
        Точечное облако: (256, 3)
    """
    vertices, faces = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    vertices = vertices[ids]
    return np.asarray(vertices)


def mesh_to_image(mesh: 'open3d.geometry.TriangleMesh', camera_distance: float = -1.8, 
                  front: list = [1, 1, 1], width: int = 500, height: int = 500, 
                  img_size: int = 128) -> 'PIL.Image.Image':
    """
    Рендерит 3D mesh в 2D изображение с заданной камеры.
    
    Создает визуализацию mesh с использованием Open3D visualizer и сохраняет
    результат как изображение. Используется для создания входных данных для
    моделей, работающих с изображениями.
    
    Алгоритм:
        1. Настройка камеры с заданными параметрами (расстояние, направление взгляда)
        2. Рендеринг mesh в буфер с высоким разрешением
        3. Масштабирование до целевого размера с антиалиасингом
    
    Args:
        mesh (open3d.geometry.TriangleMesh): 3D mesh для рендеринга.
            Должен быть валидным Open3D mesh объектом.
        camera_distance (float, optional): Расстояние камеры от объекта.
            Отрицательные значения означают камеру перед объектом.
            По умолчанию -1.8.
        front (list, optional): Направление взгляда камеры [x, y, z].
            Определяет ориентацию камеры относительно объекта.
            По умолчанию [1, 1, 1] (диагональный вид).
        width (int, optional): Ширина изображения для рендеринга в пикселях.
            Используется для высококачественного рендеринга перед масштабированием.
            По умолчанию 500.
        height (int, optional): Высота изображения для рендеринга в пикселях.
            По умолчанию 500.
        img_size (int, optional): Финальный размер изображения (квадратное).
            Изображение будет масштабировано до img_size x img_size.
            По умолчанию 128.
    
    Returns:
        PIL.Image.Image: Рендеренное изображение mesh в формате PIL Image.
            Размер изображения: img_size x img_size пикселей.
    
    Raises:
        RuntimeError: Если не удалось создать визуализатор или выполнить рендеринг.
        ValueError: Если mesh пустой или невалидный.
    
    Note:
        Функция создает временное окно визуализации Open3D, которое автоматически
        закрывается после рендеринга. Используется антиалиасинг при масштабировании
        для сохранения качества изображения.
    
    Example:
        >>> import open3d as o3d
        >>> mesh = o3d.io.read_triangle_mesh('model.stl')
        >>> image = mesh_to_image(mesh, img_size=128)
        >>> image.save('rendered_model.png')
    """
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    # Настройка камеры для рендеринга mesh с заданного ракурса
    # Используем стандартную камеру Open3D с настраиваемыми параметрами
    
    # Точка, на которую смотрит камера (центр объекта)
    lookat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    # Направление взгляда камеры (вектор от объекта к камере)
    front_array = np.array(front, dtype=np.float32)
    # Вектор "вверх" для камеры (обычно ось Y)
    up = np.array([0, 1, 0], dtype=np.float32)
    
    # Вычисляем позицию камеры: отступаем от lookat в направлении front на расстояние camera_distance
    eye = lookat + front_array * camera_distance
    
    # Вычисляем ортогональный базис для камеры:
    # right - вектор "вправо" (перпендикулярен up и front)
    right = np.cross(up, front_array)
    right /= np.linalg.norm(right)  # Нормализуем
    # true_up - скорректированный вектор "вверх" (перпендикулярен front и right)
    true_up = np.cross(front_array, right)
    
    # Формируем матрицу поворота камеры из ортогональных векторов
    rotation_matrix = np.column_stack((right, true_up, front_array)).T
    # Создаем матрицу внешних параметров камеры (extrinsic matrix)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix  # Поворот
    extrinsic[:3, 3] = -rotation_matrix @ eye  # Перевод (смещение камеры)

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = skimage.transform.resize(
        image,
        output_shape=(img_size, img_size),
        order=2,
        anti_aliasing=True,
        preserve_range=True).astype(np.uint8)

    return Image.fromarray(image)


class CadRecodeDataset(Dataset):
    """
    Датасет для загрузки и обработки 3D моделей для обучения и тестирования моделей CAD.
    
    Поддерживает несколько режимов работы:
    - 'pc': работа с точечными облаками (point clouds)
    - 'img': работа с изображениями (рендеринг mesh с разных ракурсов)
    - 'pc_img': случайный выбор между точечными облаками и изображениями
    
    Датасет автоматически загружает аннотации из .pkl файлов для train/val split
    или сканирует директорию для test split. Поддерживает аугментацию данных
    и нормализацию для различных режимов работы.
    
    Attributes:
        root_dir (str): Корневая директория с данными. Должна содержать поддиректории
            с датасетами или .pkl файлы для train/val.
        split (str): Раздел датасета. Может быть:
            - 'train': обучающий набор (загружается из train.pkl)
            - 'val': валидационный набор (загружается из val.pkl)
            - Любое другое значение: тестовый набор (сканируется директория)
        n_points (int): Количество точек в точечном облаке для режима 'pc'.
        normalize_std_pc (float): Стандартное отклонение для нормализации точечных облаков
            в режиме обучения. Используется для масштабирования координат.
        noise_scale_pc (float, optional): Масштаб шума для аугментации точечных облаков.
            Если None, аугментация не применяется. По умолчанию None.
        img_size (int): Размер изображения для режима 'img' в пикселях.
            Изображения всегда квадратные (img_size x img_size).
        normalize_std_img (float): Стандартное отклонение для нормализации mesh
            перед рендерингом изображений.
        noise_scale_img (float): Масштаб шума для изображений. Отрицательные значения
            отключают аугментацию.
        num_imgs (int): Количество изображений для режима 'img'. Может быть 1, 2 или 4.
            Изображения рендерятся с разных ракурсов и объединяются.
        mode (str): Режим работы датасета. Допустимые значения: 'pc', 'img', 'pc_img'.
        n_samples (int, optional): Максимальное количество образцов для загрузки.
            Если None, загружаются все образцы. По умолчанию None.
        ext (str, optional): Расширение файлов mesh для поиска в тестовом режиме.
            По умолчанию 'stl'.
        annotations (list): Список аннотаций с путями к mesh файлам и метаданными.
    
    Note:
        Для режима 'pc_img' каждый раз при обращении к элементу случайно выбирается
        между точечным облаком и изображением с вероятностью 50/50.
    
    Example:
        >>> dataset = CadRecodeDataset(
        ...     root_dir='/workspace/data',
        ...     split='deepcad_test_mesh',
        ...     n_points=256,
        ...     normalize_std_pc=100,
        ...     noise_scale_pc=None,
        ...     img_size=128,
        ...     normalize_std_img=200,
        ...     noise_scale_img=-1,
        ...     num_imgs=4,
        ...     mode='pc',
        ...     n_samples=50
        ... )
        >>> item = dataset[0]
        >>> print(item.keys())
        dict_keys(['point_cloud', 'description', 'file_name'])
    """
    
    def __init__(self, root_dir, split, n_points, normalize_std_pc, noise_scale_pc, img_size,
                normalize_std_img, noise_scale_img, num_imgs, mode, n_samples=None, ext='stl'):
        """
        Инициализирует датасет CadRecodeDataset.
        
        Args:
            root_dir (str): Корневая директория с данными
            split (str): Раздел датасета ('train', 'val' или название test split)
            n_points (int): Количество точек в точечном облаке
            normalize_std_pc (float): Стандартное отклонение для нормализации PC
            noise_scale_pc (float, optional): Масштаб шума для аугментации PC
            img_size (int): Размер изображения в пикселях
            normalize_std_img (float): Стандартное отклонение для нормализации mesh перед рендерингом
            noise_scale_img (float): Масштаб шума для изображений
            num_imgs (int): Количество изображений (1, 2 или 4)
            mode (str): Режим работы ('pc', 'img' или 'pc_img')
            n_samples (int, optional): Максимальное количество образцов
            ext (str, optional): Расширение файлов mesh. По умолчанию 'stl'
        
        Raises:
            FileNotFoundError: Если не найден .pkl файл для train/val split
            ValueError: Если mode имеет недопустимое значение или num_imgs не в {1, 2, 4}
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.n_samples = n_samples
        self.n_points = n_points  
        self.normalize_std_pc = normalize_std_pc
        self.noise_scale_pc = noise_scale_pc
        self.normalize_std_img = normalize_std_img
        self.noise_scale_img = noise_scale_img
        self.num_imgs = num_imgs
        self.mode = mode
        if self.split in ['train', 'val']:
            pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
            with open(pkl_path, 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            paths = os.listdir(os.path.join(self.root_dir, self.split))
            self.annotations = [
                {'mesh_path': os.path.join(self.split, f)}
                for f in paths if f.endswith('.stl')
            ]

    def __len__(self):
        """
        Возвращает количество образцов в датасете.
        
        Returns:
            int: Количество образцов. Если задан n_samples, возвращает минимум
                между n_samples и реальным количеством аннотаций.
        """
        if self.n_samples is not None:
            return min(self.n_samples, len(self.annotations))
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Получает элемент датасета по индексу.
        
        Загружает mesh файл и преобразует его в требуемый формат в зависимости
        от режима работы датасета. Для режима 'pc_img' случайно выбирает между
        точечным облаком и изображением.
        
        Args:
            index (int): Индекс элемента в датасете. Используется модульная
                арифметика для поддержки ConcatDataset.
        
        Returns:
            dict: Словарь с данными элемента. Содержит:
                - 'point_cloud' (np.ndarray, optional): Точечное облако формы (n_points, 3)
                  для режимов 'pc' и 'pc_img'
                - 'video' (list, optional): Список изображений PIL.Image для режима 'img'
                - 'description' (str): Текстовое описание задачи ('Generate cadquery code')
                - 'file_name' (str): Имя файла без расширения
                - 'answer' (str, optional): CadQuery код для режима обучения
        
        Raises:
            IndexError: Если индекс выходит за границы датасета
            FileNotFoundError: Если mesh файл не найден
            ValueError: Если mode имеет недопустимое значение
        """
        # Используем модульную арифметику для циклического доступа к аннотациям
        # Это позволяет ConcatDataset работать корректно при повторении датасета
        actual_index = index % len(self.annotations) if len(self.annotations) > 0 else 0
        item = self.annotations[actual_index]

        if self.mode == 'pc':
            input_item = self.get_point_cloud(item)
        elif self.mode == 'img':
            input_item = self.get_img(item)
        elif self.mode == 'pc_img':
            if np.random.rand() < 0.5:
                input_item = self.get_point_cloud(item)
            else:
                input_item = self.get_img(item)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        input_item['file_name'] = os.path.basename(item['mesh_path'])[:-4]

        if self.split in ['train', 'val']:
            py_path = item['py_path']
            py_path = os.path.join(self.root_dir, py_path)
            with open(py_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer

        return input_item

    def get_img(self, item):
        """
        Генерирует изображения mesh с разных ракурсов для режима 'img'.
        
        Рендерит mesh с 4 различных ракурсов (определенных векторами front)
        и объединяет их в одно или несколько изображений в зависимости от num_imgs.
        Применяет нормализацию mesh перед рендерингом для режима обучения.
        
        Args:
            item (dict): Элемент аннотации с ключом 'mesh_path'.
                Должен содержать путь к mesh файлу относительно root_dir.
        
        Returns:
            dict: Словарь с ключами:
                - 'video' (list): Список изображений PIL.Image. Количество зависит
                  от num_imgs (1, 2 или 4 изображения, объединенных в сетку)
                - 'description' (str): Текстовое описание задачи
        
        Raises:
            FileNotFoundError: Если mesh файл не найден
            ValueError: Если num_imgs не в {1, 2, 4}
            RuntimeError: Если не удалось выполнить рендеринг
        
        Note:
            Для num_imgs=1 возвращается одно изображение с ракурса [1, 1, 1].
            Для num_imgs=2 возвращается горизонтальная композиция двух изображений.
            Для num_imgs=4 возвращается сетка 2x2 с четырьмя ракурсами.
        """
        mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
        if self.split in ['train', 'val']:
            mesh.apply_transform(trimesh.transformations.scale_matrix(1 / self.normalize_std_img))
            mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(vertices)
        mesh.triangles = open3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        mesh.compute_vertex_normals()

        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        images = []
        for front in fronts:
            image = mesh_to_image(
                mesh, camera_distance=-0.9, front=front, img_size=self.img_size)
            images.append(image)
                
        images = [ImageOps.expand(image, border=3, fill='black') for image in images]
        if self.num_imgs == 1:
            images = [images[0]]
        elif self.num_imgs == 2:
            images = [Image.fromarray(np.hstack((
                np.array(images[0]), np.array(images[1])
            )))]
        elif self.num_imgs == 4:
            images = [Image.fromarray(np.vstack((
                np.hstack((np.array(images[0]), np.array(images[1]))),
                np.hstack((np.array(images[2]), np.array(images[3])))
            )))]
        else:
            raise ValueError(f'Invalid number of images: {self.num_imgs}')

        input_item = {
            'video': images,
            'description': 'Generate cadquery code'
        }
        return input_item

    def get_point_cloud(self, item):
        """
        Генерирует точечное облако из mesh для режима 'pc'.
        
        Загружает mesh, применяет аугментацию (если включена), конвертирует
        в точечное облако и нормализует координаты. Для режима обучения
        нормализация выполняется делением на normalize_std_pc, для тестирования
        применяется линейное преобразование к диапазону [-1, 1].
        
        Args:
            item (dict): Элемент аннотации с ключом 'mesh_path'.
                Должен содержать путь к mesh файлу относительно root_dir.
        
        Returns:
            dict: Словарь с ключами:
                - 'point_cloud' (np.ndarray): Точечное облако формы (n_points, 3)
                  с нормализованными координатами
                - 'description' (str): Текстовое описание задачи
        
        Raises:
            FileNotFoundError: Если mesh файл не найден
            ValueError: Если mesh пустой или невалидный
        
        Note:
            Аугментация применяется только в режиме обучения и только с вероятностью 50%.
            Нормализация различается для train/val и test split для обеспечения
            корректной работы модели на разных этапах.
        """
        mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
        mesh = self._augment_pc(mesh)
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        
        if self.split in ['train', 'val']:
            point_cloud = point_cloud / self.normalize_std_pc
        else:
            point_cloud = (point_cloud - 0.5) * 2
        
        input_item = {
            'point_cloud': point_cloud,
            'description': 'Generate cadquery code',
        }
        return input_item

    def _augment_pc(self, mesh):
        """
        Приватный метод для аугментации точечного облака добавлением шума.
        
        Добавляет гауссовский шум к вершинам mesh для увеличения разнообразия
        данных в режиме обучения. Применяется с вероятностью 50%.
        
        Args:
            mesh (trimesh.Trimesh): Mesh для аугментации. Модифицируется на месте.
        
        Returns:
            trimesh.Trimesh: Тот же mesh объект с добавленным шумом (если применен).
        
        Note:
            Шум добавляется только если noise_scale_pc не None и случайное число < 0.5.
            Модифицирует mesh на месте, не создает копию.
        """
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh


class Text2CADDataset(Dataset):
    """
    Датасет для работы с текстовыми описаниями и CadQuery кодами.
    
    Используется для обучения и тестирования моделей в текстовом режиме,
    где входными данными являются текстовые описания 3D объектов, а выходными -
    соответствующие CadQuery коды.
    
    Датасет загружает аннотации из .pkl файлов, которые содержат текстовые
    описания и пути к соответствующим CadQuery файлам.
    
    Attributes:
        root_dir (str): Корневая директория с данными. Должна содержать:
            - {split}.pkl: файл с аннотациями
            - {code_dir}/: директория с CadQuery файлами
        split (str): Раздел датасета ('train', 'val' или 'test')
        code_dir (str): Название поддиректории с CadQuery файлами.
            По умолчанию 'cadquery'.
        n_samples (int, optional): Максимальное количество образцов для загрузки.
            Если None, загружаются все образцы. По умолчанию None.
        annotations (list): Список аннотаций с текстовыми описаниями и метаданными.
    
    Example:
        >>> dataset = Text2CADDataset(
        ...     root_dir='/workspace/data/text2cad',
        ...     split='test',
        ...     code_dir='cadquery',
        ...     n_samples=100
        ... )
        >>> item = dataset[0]
        >>> print(item.keys())
        dict_keys(['description', 'file_name', 'answer'])
    """
    
    def __init__(self, root_dir, split, code_dir='cadquery', n_samples=None):
        """
        Инициализирует датасет Text2CADDataset.
        
        Args:
            root_dir (str): Корневая директория с данными
            split (str): Раздел датасета ('train', 'val' или 'test')
            code_dir (str, optional): Название поддиректории с CadQuery файлами.
                По умолчанию 'cadquery'.
            n_samples (int, optional): Максимальное количество образцов.
                По умолчанию None.
        
        Raises:
            FileNotFoundError: Если не найден .pkl файл для указанного split
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_samples = n_samples
        self.code_dir = code_dir
        pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        """
        Возвращает количество образцов в датасете.
        
        Returns:
            int: Количество образцов. Если задан n_samples, возвращает его значение,
                иначе - реальное количество аннотаций.
        """
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        """
        Получает элемент датасета по индексу.
        
        Загружает текстовое описание и соответствующий CadQuery код (если доступен).
        
        Args:
            index (int): Индекс элемента в датасете
        
        Returns:
            dict: Словарь с ключами:
                - 'description' (str): Текстовое описание 3D объекта
                - 'file_name' (str): Уникальный идентификатор образца (uid)
                - 'answer' (str, optional): CadQuery код для режима обучения
        
        Raises:
            IndexError: Если индекс выходит за границы датасета
            FileNotFoundError: Если CadQuery файл не найден (в режиме обучения)
        """
        item = self.annotations[index]

        input_item = {
            'description': item['description'],
            'file_name': item['uid']
        }

        if self.split in ['train', 'val']:
            py_path = f'{item["uid"]}.py'
            py_path = os.path.join(self.root_dir, self.code_dir, py_path)
            with open(py_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer
        return input_item
