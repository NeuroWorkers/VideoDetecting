from flask import Flask, request, jsonify, render_template
from configs.backend_config import ANALYZE_SOURCE_PATH, ANALYZE_TRANSLATED_CENTERS_PATH, ANALYZE_CENTERS_PATH, ANALYZE_POSES_PATH
from configs.server_config import SERVER_DEBUG_MODE, SERVER_PORT, SERVER_HOST
from utils.struct_util import delete_file, rename_file_without_extension
from backend.analyze_user_video import predict_pose
import os

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
app.secret_key = '2201'
app.config['UPLOAD_FOLDER'] = f'{ANALYZE_SOURCE_PATH}/'
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.mpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_message = process_video(filepath)

        return jsonify({'message': result_message}), 200

    return jsonify({'message': 'File type not allowed'}), 400


def allowed_file(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


def process_video(filepath):
    new_path = rename_file_without_extension(filepath, "1")
    class_index = predict_pose(new_path)

    class_names = ["Бег", "Присяд", "Выпады", "Наклоны",
                   "Ходьба", "Прыжки", "Пройти через турникет", "Перешагнуть через ограждение",
                   "Пролезть сквозь ограждение",
                   "Кинуть предмет", "Бросить сумку", "Постучать и заглянуть", "Постучать и зайти"]

    delete_file(new_path)
    delete_file(os.path.join(ANALYZE_POSES_PATH, "1.txt"))
    delete_file(os.path.join(ANALYZE_CENTERS_PATH, "1.txt"))
    delete_file(os.path.join(ANALYZE_TRANSLATED_CENTERS_PATH, "1.txt"))
    return class_names[class_index]


if __name__ == '__main__':
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=SERVER_DEBUG_MODE)
