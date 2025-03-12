import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD

# Google Drive'ı bağlama
from google.colab import drive
drive.mount('/content/drive')

# 1. Veri yolu ayarları
base_dir = "/content/drive/My Drive/fer2013"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# 2. Veriyi ön işleme
train_datagen = ImageDataGenerator(
rescale=1.0/255,
rotation_range=30,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
brightness_range=[0.8, 1.2],
fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(48, 48),
batch_size=128,
color_mode="grayscale",
class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(48, 48),
batch_size=128,
color_mode="grayscale",
class_mode="categorical"
)

# 3. Modeli oluştur
model = Sequential([
Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(48, 48,1)),
BatchNormalization(),
MaxPooling2D((2, 2)),
Dropout(0.3),
Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
BatchNormalization(),
MaxPooling2D((2, 2)),
Dropout(0.3),
Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
BatchNormalization(),
MaxPooling2D((2, 2)),
Dropout(0.4),
Flatten(),
Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
Dropout(0.5),
Dense(7, activation='softmax')
])

# 4. Modeli derle (SGD ile)
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9) # Öğrenme oranı ve momentum
model.compile(
optimizer=sgd_optimizer,
loss='categorical_crossentropy',
metrics=['accuracy']
)

# 5. Callbacks ekle
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# 6. Modeli eğit
history = model.fit(
train_generator,
epochs=50,
validation_data=test_generator,
callbacks=[early_stopping, lr_scheduler]
)

# 7. Modeli kaydet
model.save("/content/drive/My Drive/emotion_detection_model_optimized_sgd.h5")

# 8. Eğitim süreci görselleştirme
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# 9. Test veri seti üzerindeki doğruluk oranını hesapla
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Doğruluğu: {test_accuracy * 100:.2f}%")
