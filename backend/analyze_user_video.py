import os
import numpy as np
import joblib
from collections import Counter
from configs.backend_config import ANALYZE_POSES_PATH, ANALYZE_CENTERS_PATH, ANALYZE_SOURCE_PATH, \
    ANALYZE_TRANSLATED_CENTERS_PATH, SAVED_MODEL_PATH, SAVED_SCALER_PATH, WINDOW_STEP, WINDOW_SIZE
from backend.train_model.Analyzators.Dot_Finding import pose_detection
from backend.train_model.Analyzators.Mass_center import find_centers, translate_coordinates_to_centers
from backend.train_model.Analyzators.Processing import window_processing, read_files
from utils.checkers import check_video_file_extension
from utils.counters import count_frames
from utils.struct_util import delete_file


def load_model() -> tuple[any, any]:
    # Загрузка модели
    model = joblib.load(SAVED_MODEL_PATH)

    # Загрузка scaler
    scaler = joblib.load(SAVED_SCALER_PATH)

    return model, scaler


def process_video(new_path):
    # Детектирование поз
    extension = check_video_file_extension(new_path)
    pose_detection(1, ANALYZE_SOURCE_PATH, ANALYZE_POSES_PATH, extension)

    # Поиск центров масс
    source_filename = os.path.join(ANALYZE_POSES_PATH, "1.txt")
    output_filename = os.path.join(ANALYZE_CENTERS_PATH, "1.txt")
    find_centers(source_filename, output_filename)

    # Транслирование центров
    translate_coordinates_to_centers(ANALYZE_POSES_PATH, ANALYZE_CENTERS_PATH, ANALYZE_TRANSLATED_CENTERS_PATH, 1)

    # Формирование матрицы транслированных координат
    translated_centers_values = read_files(1, ANALYZE_TRANSLATED_CENTERS_PATH)

    # Подсчет кадров
    count_frames_in_files = count_frames(1, ANALYZE_POSES_PATH)

    # Обработка окном
    window_processed_data = window_processing(translated_centers_values, count_frames_in_files, WINDOW_SIZE,
                                              WINDOW_STEP)

    return window_processed_data


def prepare_for_predict(window_processed_data: list, scaler: any):
    scaled_data = scaler.transform(window_processed_data)
    return scaled_data


def predict_pose(new_path):
    model, scaler = load_model()
    window_processed_data = process_video(new_path)
    scaled_data = prepare_for_predict(window_processed_data, scaler)

    probabilities = model.predict_proba(scaled_data)
    predicted_classes = np.argmax(probabilities, axis=1)
    most_common_class = Counter(predicted_classes).most_common(1)[0][0]

    return most_common_class
