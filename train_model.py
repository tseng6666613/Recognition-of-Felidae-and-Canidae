import tensorflow as tf  #深度學習主套件
from tensorflow.keras.preprocessing.image import ImageDataGenerator   #圖片資料自動產生器，支援資料增強（augmentation）
from tensorflow.keras.applications import MobileNetV2                 #內建的「輕量級圖像辨識模型」，拿來做轉移學習
from tensorflow.keras.models import Model                             #模型建構用的元件
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D     
import os

# 圖片與訓練設定(定義參數（圖片大小、批次、訓練次數）)
IMAGE_SIZE = (224, 224)       # 輸入圖片大小（MobileNetV2 預設用 224x224）
BATCH_SIZE = 32               # 每批訓練圖片數量
EPOCHS = 5                    # 訓練輪數（建議初學先少跑）
DATASET_DIR = "dataset"       # 資料夾路徑

# 建立資料生成器（含資料增強）
datagen = ImageDataGenerator(
    rescale=1./255,             # 將像素值從 0~255 變成 0~1（正規化）
    validation_split=0.2,       # 訓練資料分 80%，驗證資料分 20%
    horizontal_flip=True,       # 左右翻轉
    rotation_range=15,          # 隨機旋轉 ±15 度
    zoom_range=0.1              # 隨機縮放 ±10%
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 使用 MobileNetV2 當 base model
base_model = MobileNetV2(include_top=False, input_shape=IMAGE_SIZE + (3,), weights='imagenet')
base_model.trainable = False  #不要訓練這些層（先凍結）MobileNet 「轉移學習」骨架，不用從零開始訓練。

x = base_model.output
x = GlobalAveragePooling2D()(x)  #把「特徵圖」變成「一維向量」把一張很複雜的圖片濃縮成「一行最重要的特徵數字
preds = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# 儲存模型
model.save("felidae_canidae_model.h5")
