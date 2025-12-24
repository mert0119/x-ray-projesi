"""
MedScan AI - Beyin Tümör Modeli Eğitimi
Transfer Learning ile 4 sınıflı sınıflandırma:
- glioma
- meningioma
- notumor (sağlıklı)
- pituitary
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import json

# Bozuk görüntüleri atla
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# GPU ayarları
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU bulundu: {len(gpus)} adet")
else:
    print("⚠️ GPU bulunamadı, CPU kullanılacak")

# ==========================================
# AYARLAR
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# Dataset yolu
DATASET_PATH = r"B:\ttümor"

print(f"\n📁 Dataset yolu: {DATASET_PATH}")

# Klasör yapısını kontrol et
if os.path.exists(DATASET_PATH):
    subfolders = os.listdir(DATASET_PATH)
    print(f"📂 Mevcut sınıflar: {subfolders}")
    print(f"📊 Toplam sınıf sayısı: {len(subfolders)}")
else:
    print(f"❌ Dataset klasörü bulunamadı: {DATASET_PATH}")
    exit(1)

# ==========================================
# VERİ HAZIRLIĞI
# ==========================================
print("\n🔄 Veri hazırlanıyor...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

# Eğitim verileri (4 sınıf = categorical)
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Çoklu sınıf
    subset='training',
    shuffle=True
)

# Validasyon verileri
validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\n✅ Eğitim örnekleri: {train_generator.samples}")
print(f"✅ Validasyon örnekleri: {validation_generator.samples}")
print(f"📊 Sınıflar: {train_generator.class_indices}")

# ==========================================
# MODEL OLUŞTURMA
# ==========================================
print("\n🧠 Model oluşturuluyor...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Softmax for multi-class

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"📊 Model hazır - {num_classes} sınıf için eğitilecek")

# ==========================================
# EĞİTİM
# ==========================================
print("\n🚀 Eğitim başlıyor...")

checkpoint = ModelCheckpoint(
    'brain_model_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ==========================================
# SONUÇLAR
# ==========================================
print("\n" + "="*50)
print("📊 BEYİN TÜMÖR MODELİ EĞİTİMİ TAMAMLANDI!")
print("="*50)

final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"✅ Eğitim Doğruluk: {final_acc*100:.2f}%")
print(f"✅ Validasyon Doğruluk: {final_val_acc*100:.2f}%")

# Model kaydet
model.save('brain_classifier_model.keras')
print(f"\n💾 Model kaydedildi: brain_classifier_model.keras")

# Sınıf indekslerini kaydet
with open('brain_class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print(f"💾 Sınıf indeksleri kaydedildi: brain_class_indices.json")

print("\n✅ Beyin tümör modeli başarıyla eğitildi!")
