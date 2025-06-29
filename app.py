from keras.models import load_model
loaded_model = load_model("Oka.h5")
# use model to predict
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import keras.utils as image_utils
import numpy as np

# 載入模型
# loaded_model = load_model("Oka.txt")
""""
# 載入圖片並預處理
img_path = "oka2.jpg"
img = image.load_img(img_path, target_size=(224, 224))  # 根據模型輸入尺寸調整
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # 若模型訓練時有正規化
"""
def load_and_process_image(image_path):
    if "http" in image_path:
        image_path = image_utils.get_file(origin=image_path)
    image_s = image_utils.load_img(image_path, target_size=(224, 224))
    image_s_array = image_utils.img_to_array(image_s)
    image_s_array_reshape = image_s_array.reshape(1, 224, 224, 3)
    image_forVGG16 = preprocess_input(image_s_array_reshape)
    return image_forVGG16

def make_predictions(image_path):
    image = load_and_process_image(image_path)
    predictions = loaded_model.predict(image)
    return predictions

def tell_oka(image_path):
    # 預測
    pred = loaded_model.predict(image_path)
    print("預測結果：", pred)
    if pred[0] < 18 and pred[0] > -8:
        print("It's OKa!")
    else:
        print("You are NOT OKa!")

import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def show_image():
    response = requests.get(f"https://api-data.line.me/v2/bot/message/566774554824540315/content", headers={
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
        "Authorization": "Bearer DE/9DDY8DN6Lfm23wS8AJNIEMrHZ4/aNbPfCEUZ073HuZe6G3nmowA2eCD8+owakR9wBjtPFO0m5nNZwXjOoPP0pJl+Ce6HBXw1uXjn1NLpJtwPSw//7rdnUs+vC6TIkdMSqaCWuvIQsPGU1KOPQkgdB04t89/1O/w1cDnyilFU="})
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  # 根據模型需求調整
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    
img = show_image()
tell_oka(img)