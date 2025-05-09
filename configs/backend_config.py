import os


# PROJECT_PATHS
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
RESULT_POSES_PATH = os.path.join(ROOT_PATH, "backend/train_model/Analyze_results/Poses")
RESULT_CENTERS_PATH = os.path.join(ROOT_PATH, "backend/train_model/Analyze_results/Centers")
RESULT_TRANSLATED_CENTERS_PATH = os.path.join(ROOT_PATH, "backend/train_model/Analyze_results/Translated_centers")
SOURCE_PATH = os.path.join(ROOT_PATH, "backend/train_model/Analyze_source/Source")
SAVED_MODEL_PATH = os.path.join(ROOT_PATH, "backend/saved_model/model.pkl")
SAVED_SCALER_PATH = os.path.join(ROOT_PATH, "backend/saved_model/scaler.pkl")


# PROJECT_VARIABLES
WINDOW_SIZE = 20
WINDOW_STEP = 10


# ANALYZE_PATHS
ANALYZE_POSES_PATH = os.path.join(ROOT_PATH, "backend/user_classification_data/Analyze_result/Pose")
ANALYZE_CENTERS_PATH = os.path.join(ROOT_PATH, "backend/user_classification_data/Analyze_result/Center")
ANALYZE_TRANSLATED_CENTERS_PATH = os.path.join(ROOT_PATH, "backend/user_classification_data/Analyze_result/Translated_center")
ANALYZE_SOURCE_PATH = os.path.join(ROOT_PATH, "backend/user_classification_data/Analyze_source")
