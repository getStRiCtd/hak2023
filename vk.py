import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import easyocr

img_fns = glob('Data/vk/images/*')
num = 234
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

def get_data_vk(df, img_path):
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
    sum = 0
    for i in data.keys():
        sum+= data[i]
    return sum




if __name__ == "__main__":
    img_path = img_fns[num]
    results = reader.readtext(img_path)

    df = pd.DataFrame(results, columns=['bbox','text','conf'])

    res = get_data_vk(df, img_path)
    res = sum(res)
