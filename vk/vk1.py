import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import easyocr

img_fns = glob('Data/vk/image/*')
num = 5
reader = easyocr.Reader(['en', 'ru'], gpu = True)
REGEXP = r'^[1-9]*[ ]?[Дд]ру|[чн]ик[иа(ов)]*'


def convert_to_dict(string: str) -> None:
  out = string.split() 
  if len(out) == 2:
    try:
      out = {out[1]: int(out[0])}
      return out
    except:
      pass

    try:
      out = {out[0]: int(out[1])}
      return out
    except:
      return None


def find(df, bbox:list) -> list:
  """
  Расширяем область поиска
  """
  bboxes = df.bbox.tolist()
  text = df.text.tolist()
  for i in range(len(bboxes)):
   
    if (-30 < bbox[0][0] - bboxes[i][0][0] < 40) and  (0 < bbox[0][1]-bboxes[i][0][1] < 100) and text[i].isdigit():
      return int(text[i])
  else:
    return None

def convert_to_dict(string: str) -> None:
  out = string.split() 
  if len(out) == 2:
    try:
      out = {out[1]: int(out[0])}
      return out
    except:
      pass

    try:
      out = {out[0]: int(out[1])}
      return out
    except:
      return None

def get_data_vk(df, img_path) -> dict:
  """
  Считывает полученные слова со скриншота и ищет среди них следующие: [подписчики, друзья] по регулярному выражению
  если нет возмоности счесть метрику, вызывает функцию find.
  Функция find находит ближайшее к названию метрики число на скриншоте и считает его значеним этой метрики
  """
  use_to = pd.DataFrame(df['text'].str.lower())
  use_to['bbox'] = df['bbox']
  use_to = use_to.loc[use_to['text'].str.contains(REGEXP)]
  tmp = use_to['text'].tolist()
  bboxes = use_to['bbox'].tolist()
  out = {}
    
  for i in range(len(tmp)):
      g = convert_to_dict(tmp[i])
      if g:
        out.update(g)
      else:
        check = find(df, bboxes[i])
        if check:
          out.update({tmp[i]:check})

  return out

def sum(data:dict) -> int:
  """
  Возвращает сумму подписчиков + друзей
  """
  sum = 0
  for i in data.keys():
      sum+= data[i]
  return sum




if __name__ == "__main__":
  print(img_fns)

  img_path = img_fns[num]
  results = reader.readtext(img_path)
  df = pd.DataFrame(results, columns=['bbox','text','conf'])

  res = get_data_vk(df, img_path)
  res = sum(res)
# ----------------------------------------------------------------------------------
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