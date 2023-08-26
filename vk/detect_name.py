from roboflow import Roboflow
import cv2 as cv
import easyocr

rf = Roboflow(api_key="fKNgN4TLK6ZSRlRHPdZZ")
project = rf.workspace().project("hak2023_2")
model = project.version(1).model

IMG_PATH = ''
nums = 2
a = model.predict(IMG_PATH, confidence=40, overlap=30).json()

def read_name_id_vk(data:dict, img_path) -> None:
  print(data)
  if not data['predictions']:
    return
  if data['predictions'][0]['confidence'] < 0.5:
    print("uncertainty")
    return
  bbox = data['predictions'][0]
  print(bbox['x'], bbox['x']+400)
  img = cv.imread(img_path)
  x = bbox['x'] - bbox['width']//2
  to_x = bbox['x'] + bbox['width']//2
  
  y = bbox['y'] - bbox['height']//2
  to_y = bbox['y'] + bbox['height']//2

  cropped_img = img[y: to_y, x:to_x]
  cv.imwrite("d.png", img)
  cv.imwrite("screen.png", cropped_img)
  
  results = reader.readtext(cropped_img)
  out = results[0][1]
  return out


read_name_id_vk(a, IMG_PATH)