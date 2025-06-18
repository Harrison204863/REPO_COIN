import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from datetime import datetime
#conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
#pip install tensorflow-gpu==2.10.0
#pip install numpy==1.22.1 scipy==1.9 pandas==1.5 matplotlib==3.5 seaborn scikit-learn==1.1 Pillow

# 基礎路徑
base_dir = r"E:\Study\IMG_data\world_coins\coins_min\data"

# 各目錄路徑
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# 標籤檔案路徑（假設與 data 目錄同層級）
label_dir = os.path.join(os.path.dirname(base_dir), "cat_to_name_min.json")
# 設定路徑
# train_dir = r"E:\Study\IMG_data\world_coins\coins_min\data\train"
# validation_dir = r"E:\Study\IMG_data\world_coins\coins_min\data\validation"
# test_dir = r"E:\Study\IMG_data\world_coins\coins_min\data\test"
# label_dir = r"E:\Study\IMG_data\world_coins\coins_min\cat_to_name_min.json"
model_dir = r"E:\Study\PyCharm\world_coin\coins_model_min_0618_01.tflite"

# ///GPU ///
# 顯示可用的 GPU 裝置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 限制第一張 GPU 使用最多 4096 MB 記憶體
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s).")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Using CPU.")

# 創建結果儲存目錄
results_dir = os.path.join(os.path.dirname(model_dir), "training_results_0618_01")
os.makedirs(results_dir, exist_ok=True)

# 載入類別標籤
with open(label_dir, 'r') as f:
    cat_to_name = json.load(f)
class_names = list(cat_to_name.values())

# 資料增強設定
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# 生成資料流
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True)
val_generator = val_datagen.flow_from_directory(
    validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)
test_generator = val_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# 模型定義 (使用 MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20
)


# ==================== 評估與儲存結果 ====================
def save_plots_and_reports():
    # 1. 繪製訓練曲線
    plt.figure(figsize=(12, 6))

    # Loss 曲線
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy 曲線
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'))
    plt.close()

    # 2. 生成分類報告 (Precision/Recall/F1)
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(results_dir, 'classification_report.csv'), index=True)

    # 3. 混淆矩陣熱力圖
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 儲存訓練日誌
    with open(os.path.join(results_dir, 'training_log.txt'), 'w') as f:
        f.write(f"Training completed at {datetime.now()}\n\n")
        f.write("===== Classification Report =====\n")
        f.write(classification_report(y_true, y_pred_classes, target_names=class_names))
        f.write("\n\n===== Model Summary =====\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))


# 執行儲存函數
save_plots_and_reports()

# 儲存模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(model_dir, 'wb') as f:
    f.write(tflite_model)

print(f"所有結果已儲存至: {results_dir}")