# -*- coding: utf-8 -*-
"""WatermarksCNN_con_ADAM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1T0T04bIMs3ZPED8m_O38pJoYPa1emHDp
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/MyDrive/WaterMarkDetector"
!ls

import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def preprocess_images(directory):
    """
    Función para preprocesar las imágenes en un directorio.
    Intenta abrir y guardar cada imagen para eliminar aquellas que no se puedan abrir.

    Args:
    - directory: Directorio que contiene las imágenes a procesar.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                # Intenta abrir y guardar la imagen
                img.save(img_path)
            except Exception as e:
                # Si hay un error al abrir la imagen, elimínala
                os.remove(img_path)

# Directorio donde se encuentran las imágenes
train_dir = 'watermarks/train'
validation_dir = 'watermarks/validation'

# Preprocesar imágenes
preprocess_images(train_dir)
preprocess_images(validation_dir)

# Generador de datos de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=64,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 100),
    batch_size=64,
    class_mode='binary'
)


# Obtener un solo lote de imágenes y etiquetas
batch_x, batch_y = next(train_generator)

# Imprimir la forma de batch_x y batch_y
print(batch_x.shape)
print(batch_y.shape)

# Visualizar las primeras 10 imágenes
plt.figure()
f, axarr = plt.subplots(1, 10, figsize=(30, 4))
for i in range(10):
    axarr[i].imshow(batch_x[i])
    axarr[i].axis("off")
    axarr[i].set_title(batch_y[i])

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Aumentar la complejidad del modelo
model = Sequential([
    # Capa convolucional con 64 filtros y función de activación ReLU
    Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),  # Capa de pooling

    # Capa convolucional con 128 filtros y función de activación ReLU
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  # Capa de pooling

    # Capa convolucional con 256 filtros y función de activación ReLU
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  # Capa de pooling

    # Capa convolucional con 512 filtros y función de activación ReLU
    Conv2D(512, (3, 3), activation='relu'),

    Flatten(),  # Aplanar la salida para conectarla a capas densas

    # Capa densa con 512 neuronas y función de activación ReLU
    Dense(512, activation='relu'),

    # Capa densa de salida con activación sigmoide para clasificación binaria
    Dense(1, activation='sigmoid')
])

# Compilar el modelo con el optimizador Adam y una tasa de aprendizaje reducida
opt = Adam(learning_rate=0.0005)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),

)

import matplotlib.pyplot as plt

# Graficar resultados de precisión (accuracy) y pérdida (loss)

# Obtener datos de precisión y pérdida del historial de entrenamiento
acc = history.history['accuracy']
loss = history.history['loss']

# Crear una lista de épocas para el eje x
epochs = range(1, len(acc) + 1)

# Graficar precisión (accuracy) en el conjunto de entrenamiento
plt.plot(epochs, acc, 'bo', label='train accuracy')
plt.title('train acc')  # Título del gráfico de precisión
plt.legend()  # Mostrar la leyenda

# Crear una nueva figura para graficar la pérdida (loss)
plt.figure()

# Graficar pérdida (loss) en el conjunto de entrenamiento
plt.plot(epochs, loss, 'bo', label='training loss')
plt.title('train loss')  # Título del gráfico de pérdida
plt.legend()  # Mostrar la leyenda

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Obtener las predicciones del modelo para los datos de entrenamiento y validación
y_pred_train = model.predict(train_generator)
y_pred_validation = model.predict(validation_generator)

# Convertir las predicciones en etiquetas binarias (0 o 1)
y_pred_train_binary = (y_pred_train > 0.5).astype(int)
y_pred_validation_binary = (y_pred_validation > 0.5).astype(int)

# Obtener las etiquetas verdaderas para los datos de entrenamiento y validación
y_true_train = train_generator.classes
y_true_validation = validation_generator.classes

# Calcular la matriz de confusión para los datos de entrenamiento y validación
cm_train = confusion_matrix(y_true_train, y_pred_train_binary)
cm_validation = confusion_matrix(y_true_validation, y_pred_validation_binary)

# Visualizar las matrices de confusión con seaborn
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Datos de Entrenamiento')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.subplot(1, 2, 2)
sns.heatmap(cm_validation, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Datos de Validación')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.tight_layout()
plt.show()

# Definir el nombre del archivo donde se guardará el modelo
model_filename = 'modelo_optimizado_cnn_watermarks.h5'

# Guardar el modelo en el archivo especificado
model.save(model_filename)

# Verificar si el archivo se ha guardado correctamente
if os.path.exists(model_filename):
    print(f"Modelo guardado correctamente en {model_filename}")
else:
    print("Error al guardar el modelo")

import os

# Definir el nombre del archivo donde se guardará el modelo
model_filename = 'modelo_optimizado_cnn_watermarks.h5'

# Obtener la ruta completa al archivo
full_path = os.path.abspath(model_filename)

print(f"Modelo guardado en: {full_path}")