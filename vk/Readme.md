# Метрки в VK
Используется схожий алгоритм. Тем не меннее, пользователи присылают скрины из рализличных версий приложения (мобильное для андроида и ios, браузерное мобильное и пк).
Часто скриншот не содержит информации и имени пользователя. Имеется только id. Поэтому было обучена сеть YOLO v8 для нахождения информации о имени и id аккаунта, где это возможно.

## Файлы
[**detect_features.py**](https://github.com/getStRiCtd/hak2023/blob/main/vk/detect_features.py) - определение метрик с помощью easyocr.  
[**detect_name.py**](https://github.com/getStRiCtd/hak2023/blob/main/vk/detect_name.py) - модели и соотвуствующие методы.  
[**vk.py**](https://github.com/getStRiCtd/hak2023/blob/main/vk/vk.py) - основной файл, из которого происходит работа с программой.  
[**vk1.py**](https://github.com/getStRiCtd/hak2023/blob/main/vk/vk1.py) - черновые функции.
