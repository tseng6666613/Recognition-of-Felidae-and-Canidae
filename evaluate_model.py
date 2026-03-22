import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 載入模型
model = tf.keras.models.load_model('felidae_canidae_model.h5')

# 類別順序（依照原始 dataset 順序）
class_names = sorted(os.listdir('dataset'))

# 加上 unknown 處理後的所有可能預測
final_classes = class_names + ['unknown']

y_true = []
y_pred = []
confidences = []

# 根據真實標籤資料夾來做
test_base = "test_dataset"

for class_folder in os.listdir(test_base):
    folder_path = os.path.join(test_base, class_folder)
    if not os.path.isdir(folder_path):
        continue
    image_paths = glob(os.path.join(folder_path, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_path in image_paths:
        # 處理圖片
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 預測
        predictions = model.predict(img_array, verbose=0)
        confidence = np.max(predictions)
        predicted_class = class_names[np.argmax(predictions)]

        if confidence < 0.5:
            final_label = "unknown"
        elif predicted_class == "none":
            final_label = "none"
        else:
            final_label = predicted_class

        y_true.append(class_folder)
        y_pred.append(final_label)
        confidences.append(confidence)

# 產出報告
print("\n📊 準確率報告：")
print(classification_report(y_true, y_pred, labels=final_classes, zero_division=0))

# 混淆矩陣文字版
print("\n🧩 混淆矩陣：")
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
print(cm)

# ✅ 可視化混淆矩陣（熱力圖）
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=final_classes, yticklabels=final_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("\n✅ 混淆矩陣圖片已儲存為 confusion_matrix.png")

# ✅ 預測結果表格輸出
df = pd.DataFrame({
    'TrueLabel': y_true,
    'Predicted': y_pred,
    'Confidence': confidences
})
df.to_csv("prediction_report.csv", index=False)
print("\n✅ 預測結果已儲存為 prediction_report.csv")


