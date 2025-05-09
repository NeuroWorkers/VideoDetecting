import os
import numpy as np
from utils.counters import count_video_files, count_frames
from configs.backend_config import SOURCE_PATH, RESULT_POSES_PATH, RESULT_CENTERS_PATH, RESULT_TRANSLATED_CENTERS_PATH, WINDOW_STEP, WINDOW_SIZE
from backend.train_model.Analyzators.Dot_Finding import pose_detection
from backend.train_model.Analyzators.Mass_center import find_centers, translate_coordinates_to_centers
from backend.train_model.Analyzators.Processing import not_mixed_samples_train, window_processing, read_files


def get_model():
    # 1
    video_base_size = count_video_files(SOURCE_PATH)
    count_frames_in_files = []

    # 2
    # pose_detection(video_base_size, SOURCE_PATH, RESULT_POSES_PATH)

    # 3
    for i in range(video_base_size):
        source_filename = os.path.join(RESULT_POSES_PATH, f"{str(i + 1)}.txt")
        output_filename = os.path.join(RESULT_CENTERS_PATH, f"{str(i + 1)}.txt")
        find_centers(source_filename, output_filename)

    # 4
    count_frames_in_files = count_frames(video_base_size, RESULT_POSES_PATH)

    translate_coordinates_to_centers(RESULT_POSES_PATH, RESULT_CENTERS_PATH, RESULT_TRANSLATED_CENTERS_PATH,
                                     video_base_size)
    translated_centers_values = read_files(video_base_size, RESULT_TRANSLATED_CENTERS_PATH)
    window_processed_data = window_processing(translated_centers_values, count_frames_in_files, WINDOW_SIZE, WINDOW_STEP)

    # 5
    labels = np.zeros(len(window_processed_data), dtype=int)

    labels_names = [
        "Бег", "Присяд", "Выпады", "Наклоны",
        "Ходьба", "Прыжки", "Пройти через турникет", "Перешагнуть через ограждение",
        "Пролезть сквозь ограждение", "Кинуть предмет", "Бросить сумку",
        "Постучать и заглянуть", "Постучать и зайти"
    ]

    ranges_classes = [
        (0, 29, 0),
        (29, 39, 1),
        (39, 49, 2),
        (49, 59, 3),
        (59, 109, 4),
        (109, 129, 5),
        (129, 149, 6),
        (149, 169, 7),
        (169, 190, 8),
        (190, 208, 9),
        (208, 223, 10),
        (223, 228, 11),
        (228, 236, 12)
    ]

    start_index = 0

    for start_file, end_file, class_label in ranges_classes:
        for idx in range(start_file, end_file):
            length = int((count_frames_in_files[idx] - WINDOW_SIZE) / WINDOW_STEP) + 1
            labels[start_index: start_index + length] = class_label
            start_index += length

    # 6
    print(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)
    print("Уникальные классы:", unique_classes)
    print("Количество примеров для каждого класса:", counts)
    not_mixed_samples_train(window_processed_data, labels, labels_names, "RandomForest", count_frames_in_files)
