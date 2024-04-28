# -*- coding: utf-8 -*-
"""WatermarksCNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CyMPRlOHCaUwXKVUni7r2tAFJyDWVx-K
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/MyDrive/WaterMarkDetector/"
!ls

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

# Directorio de los datos de entrenamiento
train_dir = 'watermarks/train'
validation_dir = 'watermarks/validation'


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

# Preprocesar imágenes
preprocess_images(train_dir)
preprocess_images(validation_dir)


# Generador de datos de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Rango de rotación aleatoria de hasta 20 grados
    zoom_range=0.2,  # Rango de zoom aleatorio
    width_shift_range=0.2,  # Rango de desplazamiento horizontal aleatorio
    height_shift_range=0.2,  # Rango de desplazamiento vertical aleatorio
    horizontal_flip=True,  # Volteo horizontal aleatorio
    vertical_flip=True  # Volteo vertical aleatorio
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Rango de rotación aleatoria de hasta 20 grados
    zoom_range=0.2,  # Rango de zoom aleatorio
    width_shift_range=0.2,  # Rango de desplazamiento horizontal aleatorio
    height_shift_range=0.2,  # Rango de desplazamiento vertical aleatorio
    horizontal_flip=True,  # Volteo horizontal aleatorio
    vertical_flip=True  # Volteo vertical aleatorio
)

# Crear generador de datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary'
)

# Crear generador de datos de validación
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 100),
    batch_size=32,
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

import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear el modelo CNN
model = Sequential([
    # Capa convolucional con 32 filtros de 3x3 y función de activación ReLU
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),

    # Capa de pooling para reducir las dimensiones de la salida
    MaxPooling2D((2, 2)),

    # Segunda capa convolucional con 64 filtros de 3x3 y función de activación ReLU
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Tercera capa convolucional con 128 filtros de 3x3 y función de activación ReLU
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  # Capa de pooling

    # Aplanar la salida para pasarla a capas densas
    Flatten(),
    # Capa densa con 512 neuronas y función de activación ReLU
    Dense(512, activation='relu'),
    # Capa de salida con 1 neurona y función de activación sigmoide para clasificación binaria
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Entrenamiento del modelo
history = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
)

# Graficar resultados
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='train accuracy')
plt.title('train acc')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.title('train loss')
plt.legend()

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
model_filename = 'modelo_cnn_watermarks1.h5'

# Guardar el modelo en el archivo especificado
model.save(model_filename)

# Verificar si el archivo se ha guardado correctamente
if os.path.exists(model_filename):
    print(f"Modelo guardado correctamente en {model_filename}")
else:
    print("Error al guardar el modelo")

import os

# Definir el nombre del archivo donde se guardará el modelo
model_filename = 'modelo_cnn_watermarks1.h5'

# Obtener la ruta completa al archivo
full_path = os.path.abspath(model_filename)

print(f"Modelo guardado en: {full_path}")