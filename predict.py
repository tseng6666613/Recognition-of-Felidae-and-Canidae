import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# 載入模型
model = tf.keras.models.load_model('felidae_canidae_model.h5')

# 類別名稱（按訓練時自動排序）
class_names = sorted(os.listdir('dataset'))

def predict_image(img_path):
    # 讀取圖片並預處理
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 模型預測
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    # 加入 unknown 與 none 判斷
    if confidence < 0.5:
        return "unknown", confidence
    elif predicted_class == "none":
        return "none", confidence
    else:
        return predicted_class, confidence

# 測試圖片（根據修改圖片路徑）
test_image_path = test_image_path = "D:/Recognition of Felidae and Canidae/test/000008.jpg"

label, score = predict_image(test_image_path)
print(f"預測結果：{label}（信心值：{score:.2f}）")
