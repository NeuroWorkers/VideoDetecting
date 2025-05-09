import os.path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from configs.backend_config import WINDOW_SIZE, WINDOW_STEP, SAVED_SCALER_PATH, SAVED_MODEL_PATH
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# N / 2 + 1


# Чтение из файлов с описанием видеозаписей в один общий массив numbers
# Чтение из файлов с описанием видеозаписей в один общий массив numbers
def read_files(video_base_size: int, source_path: str) -> list[list[float]]:
    """
    Читает построчно файл + парсит номера строка

    Args:
        video_base_size (int): Размер базы данных с видеозаписями.
        source_path (str): Путь к базе данных с центрами.
    Returns:
        list[list[float]] - Сформированная матрица из всех файлов (последовательно).
    """
    file_values = []

    for file_index in range(0, video_base_size):
        path = os.path.join(source_path, f"{file_index + 1}.txt")

        with open(path, 'r') as file:
            data = file.readlines()
        for line in data:
            file_values.append([float(num) for num in line.split()[1:]])

    return file_values


# Создание общей матрицы, в которой хранится обработка скользящим окном по массиву numbers
def window_processing(file_values: list, count_frames_in_files: list[int], window_size: int = WINDOW_SIZE, window_step: int = WINDOW_STEP) -> list[np.array]:
    """
    Обработка скользящим окном

    Args:
        file_values (list): Сформированная матрица из файлов.
        count_frames_in_files (list[int]): Количество кадров в каждом видеофрагменте.
        window_size (int): Размер окна.
        window_step (int): Шаг окна.
    Returns:
        list[np.array] - Сформированная матрица в результате обработки скользящим окном.
    """
    result = []
    processing_counter = 0
    current_video_counter = count_frames_in_files[processing_counter]
    current_video_processing_flag = False

    for i in range(0, len(file_values) + 1 - window_size, window_step):
        if current_video_counter < i < current_video_counter + window_size / window_step:
            current_video_processing_flag = True
            continue

        if current_video_processing_flag:
            processing_counter += 1
            current_video_processing_flag = False
            current_video_counter = count_frames_in_files[processing_counter]

        window_matrix = np.array(file_values[i:i + window_size])
        result.append(window_matrix.flatten())

    return result


# Обучение на тренировочной выборке и предсказание на тестовой с помощью: SVC, KNN, DecisionTree, RandomForest
# Данные не обрабатываются предварительно, то есть описание одной видеозаписи попадает в обе выборки
def mixed_samples_train(window_processed_data: list[np.array], labels: np.array, labels_names: list[str], classifier_type: str):
    """
    Классификация на выборках без предварительной обработки.

    Args:
        window_processed_data (list[np.array]): Сформированная в результате обработки скользящим окном матрица.
        labels (np.array): Массив меток.
        labels_names (list[str): Расшифровка меток.
        classifier_type (str): Тип классификатора.
    Returns:
        None.
    """
    x_train, x_test, y_train, y_test = train_test_split(window_processed_data, labels, test_size=0.2, random_state=None)

    # Масштабирование данных
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Обучение модели
    model = None
    if classifier_type == "SVC":
        print("Модель SVC без предварительной обработки")
        model = SVC(kernel='leaf10', C=1, gamma='scale')
    elif classifier_type == "KNN":
        print("Модель KNN без предварительной обработки")
        model = KNeighborsClassifier(n_neighbors=5)
    elif classifier_type == "DecisionTree":
        print("Модель DecisionTree без предварительной обработки")
        model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1)
    elif classifier_type == "RandomForest":
        print("Модель RandomForest без предварительной обработки")
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    else:
        raise ValueError(f"Неподдерживаемый тип классификатора: {classifier_type}")

    model.fit(x_train, y_train)

    # Предсказание
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(x_test)):
        predicted_class = labels_names[y_pred[i]]
        true_class = labels_names[y_test[i]]

        # print(f"Окно {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")


# Обучение на тренировочной выборке и предсказание на тестовой с помощью: SVC, KNN, DecisionTree, RandomForest
# Данные обрабатываются предварительно(группируются), то есть описание одной видеозаписи принадлежит только одной из выборок
def not_mixed_samples_train(window_processed_data: list[np.array], labels: np.array, labels_names: list[str], classifier_type: str, count_frames_in_files: list[int]):
    scaler = StandardScaler()

    x_train = scaler.fit_transform(window_processed_data)
    model = KNeighborsClassifier(n_neighbors=1)

    model.fit(x_train, labels)

    joblib.dump(model, SAVED_MODEL_PATH)
    joblib.dump(scaler, SAVED_SCALER_PATH)
