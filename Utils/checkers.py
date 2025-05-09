import os


def get_video_extensions():
    return {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.mpeg'}


def check_video_file_extension(file_path):
    _, ext = os.path.splitext(file_path)

    if ext.lower() in get_video_extensions():
        return ext.lower()
    else:
        return ext.lower()