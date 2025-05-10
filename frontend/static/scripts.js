document.getElementById('uploadButton').addEventListener('click', function() {
    // Открываем диалог выбора файла
    document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0]; // Получаем выбранный файл

    if (file) {
        const videoPlayer = document.getElementById('videoPlayer');
        const fileURL = URL.createObjectURL(file);

        // Отображаем видео
        videoPlayer.src = fileURL;
        videoPlayer.style.display = 'block';
        videoPlayer.play(); // Автоматически начинаем воспроизведение (по желанию)

        const button = document.getElementById('uploadButton')
        button.style.marginTop = '5px';
        button.style.marginBottom = '5px';

        const container = document.getElementById('videoContainer')
        container.style.height = '95%';

        // Создаем FormData и отправляем файл на сервер
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('responseText').value = data.message; // Отображение ответа от сервера
        })
        .catch(error => console.error('Error:', error));
    } else {
        alert('Пожалуйста, выберите видеофайл для загрузки.');
    }
});
