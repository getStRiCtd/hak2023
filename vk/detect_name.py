from roboflow import Roboflow
import easyocr
import detect_features
from detect_features import reader
import cv2 as cv

"""
Содержит модель и функцию поиска имени аккаунта (или id)
read_name_id_vk возвращает либо имя, либо id аккаунта
Используется fine-tuned модель MS COCO, обученная с помощью сервиса Roboflow (нужен импорт модуля и API_KEY для доступа к ней)
Функция read_name_id_vk возвращает имя аккаунта либо его id (в зависимости что видно на скрине)
"""

rf = Roboflow(api_key="fKNgN4TLK6ZSRlRHPdZZ")
project = rf.workspace().project("hak2023_2")
model = project.version(1).model

IMG_PATH = ''
nums = 2


def read_name_id_vk(data:dict, img_path) -> str:
  """
   Функция read_name_id_vk возвращает имя аккаунта либо его id (в зависимости что видно на скрине)

  """

  if not data['predictions']:
    return
  
  if data['predictions'][0]['confidence'] < 0.5:
    print("uncertainty")
    return
  
  bbox = data['predictions'][0]
  img = cv.imread(img_path)

  x = bbox['x'] - bbox['width']//2
  to_x = bbox['x'] + bbox['width']//2
  y = bbox['y'] - bbox['height']//2
  to_y = bbox['y'] + bbox['height']//2

  cropped_img = img[y: to_y, x:to_x]

  results = reader.readtext(cropped_img)
  out = results[0][1]
  return out


if __name__ == "__main__":
    a = model.predict(IMG_PATH, confidence=40, overlap=30).json()
    read_name_id_vk(a, IMG_PATH)