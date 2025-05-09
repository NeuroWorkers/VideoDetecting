import os


def delete_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            print(f"Файл '{file_path}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка при удалении файла: {e}")


def rename_file_without_extension(current_path, new_name):
    if not os.path.isfile(current_path):
        print(f"Файл не найден: {current_path}")
        return

    directory = os.path.dirname(current_path)
    _, extension = os.path.splitext(current_path)

    new_path = os.path.join(directory, new_name + extension)

    try:
        os.rename(current_path, new_path)
    except Exception as e:
        print(f"Ошибка при переименовании файла: {e}")

    return new_path
