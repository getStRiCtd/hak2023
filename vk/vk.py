import detect_features
import detect_name

import pandas as pd

"""
detect_features: img_fns, reader = easyocr.Reader(['en', 'ru'], gpu = True)
detect_name: 
"""

img_path = detect_features.img_fns[detect_features.num]
results = detect_features.reader.readtext(img_path)


df = pd.DataFrame(results, columns=['bbox','text','conf'])

res = detect_features.get_data_vk(df, img_path)
res = detect_features.sum(res)

a = detect_name.model.predict(img_path, confidence=40, overlap=30).json()
name =  detect_name.read_name_id_vk(a, img_path)
print(name, res)