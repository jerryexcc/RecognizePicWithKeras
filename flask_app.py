import os
import requests
from flask import Flask, request, render_template, jsonify
from keras.applications.vgg16 import preprocess_input
import keras.utils as image_utils
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import keras.utils as image_utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow import keras
from io import BytesIO
from PIL import Image
from keras.models import load_model

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Oka.h5")
loaded_model = load_model(file_path)

def show_image(id):
    response = requests.get(f"https://api-data.line.me/v2/bot/message/{id}/content", headers={
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
        "Authorization": "Bearer DE/9DDY8DN6Lfm23wS8AJNIEMrHZ4/aNbPfCEUZ073HuZe6G3nmowA2eCD8+owakR9wBjtPFO0m5nNZwXjOoPP0pJl+Ce6HBXw1uXjn1NLpJtwPSw//7rdnUs+vC6TIkdMSqaCWuvIQsPGU1KOPQkgdB04t89/1O/w1cDnyilFU="})
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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

@app.route("/image", methods=['POST'])
def per_image():
    if request.is_json:
        data = request.json
        received_id = data.get('id')
        img = show_image(received_id)
        if pred[0] < 18 and pred[0] > -8:
            return jsonify({
            "status": "success",
            "predict": "True"
        }), 200
        else:
            return jsonify({
            "status": "success",
            "predict": "False"
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": "請求必須是 JSON 格式"
        }), 400

@app.route("/test")
def test():
    return jsonify({
        "status": "success",
        "message": "ID 已成功接收！"
    })


if __name__ == "__main__":
    app.run()
