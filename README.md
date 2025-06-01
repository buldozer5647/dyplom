Щоб запустити проєкт, після клонування проєкту запустіть наступні команди в консольному вікні:
- conda create --name my-env --file requirements.txt
- conda activate my-env

Якщо відсутній дистрибутив Anaconda
- python -m venv env
- env\Scripts\activate
- pip install -r requirements.txt

Після цього можна запускати відповідні програми. Варто зауважити, що модель YOLO розміром extra large (x) була навмисне не додана в репозиторій, бо займає більше ніж 100мб пам'яті.
