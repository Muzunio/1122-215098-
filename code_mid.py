import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 訓練集、測試集及標籤
train_dir = "./img/train/"
train_labels_path = "./data/train_labels.csv"
validation_dir = "./img/validation/"
validation_labels_path = "./data/validation_labels.csv"
test_dir = "./img/test/"
test_labels_path = "./data/test_labels.csv"

train_labels_df = pd.read_csv(train_labels_path)
validation_labels_df = pd.read_csv(validation_labels_path)
test_labels_df = pd.read_csv(test_labels_path)

train_images = []
validation_images = []
test_images = []

batch_size = 128
num_classes = 36

# 處理圖片與其對應的標籤
for idx, row in train_labels_df.iterrows():
    img = tf.keras.preprocessing.image.load_img(train_dir + row['filename'], target_size=(32, 32))
    img = tf.keras.preprocessing.image.img_to_array(img)
    train_images.append(img)

for idx, row in validation_labels_df.iterrows():
    img = tf.keras.preprocessing.image.load_img(validation_dir + row['filename'], target_size=(32, 32))
    img = tf.keras.preprocessing.image.img_to_array(img)
    validation_images.append(img)

for idx, row in test_labels_df.iterrows():
    img = tf.keras.preprocessing.image.load_img(test_dir + row['filename'], target_size=(32, 32))
    img = tf.keras.preprocessing.image.img_to_array(img)
    test_images.append(img)

# 正規化
train_images = np.array(train_images) / 255.0
validation_images = np.array(validation_images) / 255.0
test_images = np.array(test_images) / 255.0

# 轉為 NumPy 數組
train_labels = np.array(train_labels_df['label'])
validation_labels = np.array(validation_labels_df['label'])
test_labels = np.array(test_labels_df['label'])

all_labels = set(train_labels) | set(test_labels)

# 建立字典
label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}

train_labels_mapped = [label_map[label] for label in train_labels]
validation_labels_mapped = [label_map[label] for label in validation_labels]
test_labels_mapped = [label_map[label] for label in test_labels]

# 轉為 one-hot encoding
train_labels_encoded = tf.keras.utils.to_categorical(train_labels_mapped, num_classes)
validation_labels_encoded = tf.keras.utils.to_categorical(validation_labels_mapped, num_classes)
test_labels_encoded = tf.keras.utils.to_categorical(test_labels_mapped, num_classes)

model = models.Sequential()
model.add(layers.Conv2D(128, (2, 2), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))



model.summary()

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(train_images, train_labels_encoded, batch_size=batch_size, epochs=30, validation_data=(validation_images, validation_labels_encoded), shuffle=True)


# 繪製準確率曲線
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Train History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.3, 1])
plt.legend(loc='lower right')
plt.show()

# 繪製損失曲線
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# 評估模型在測試集的準確率
test_loss, test_acc = model.evaluate(test_images, test_labels_encoded, verbose=2)
print("Test Accuracy:", test_acc)
