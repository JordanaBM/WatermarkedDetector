# -*- coding: utf-8 -*-
"""LoadSavedModel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16yew1HL4VN3rXdzOj9QkVRo0J1sa2tXh
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd "//content/drive/MyDrive/WaterMarkDetector"
!pwd

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directorio de los datos de prueba
test_dir = 'watermarks/test'

# Cargar el modelo desde el archivo
model = load_model('modelo_cnn_watermarks1.h5')

# Crear generador de datos de prueba
test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Rango de rotación aleatoria de hasta 20 grados
    zoom_range=0.2,  # Rango de zoom aleatorio
    width_shift_range=0.2,  # Rango de desplazamiento horizontal aleatorio
    height_shift_range=0.2,  # Rango de desplazamiento vertical aleatorio
    horizontal_flip=True,  # Volteo horizontal aleatorio
    vertical_flip=True  # Volteo vertical aleatorio
    )

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))

print(f'Loss en el conjunto de prueba: {loss}')
print(f'Accuracy en el conjunto de prueba: {accuracy}')

for i in range(0, 100, 10):  # Incrementar de 10 en 10 hasta llegar a 100
    # Crear una figura con 10 subplots
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

    # Iterar sobre cada subplot y mostrar una imagen con su predicción
    for j in range(10):
        # Obtener la imagen y la etiqueta
        image = test_generator[i + j][0][0]  # Obtenemos el primer (y único) elemento del lote
        label = test_generator[i + j][1][0]

        # Obtener la predicción del modelo
        prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]
        class_name = 'watermark' if prediction >= 0.5 else 'no watermark'

        # Mostrar la imagen con su predicción en el subplot correspondiente
        row = j // 5  # Calcular la fila del subplot
        col = j % 5  # Calcular la columna del subplot
        axs[row, col].imshow(image)
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Predicción: {class_name}, Real: {"watermark" if label == 1 else "no watermark"}')

    plt.tight_layout()
    plt.show()

import seaborn as sns
import pandas as pd

# Obtener las etiquetas reales y las predicciones
y_true = labels
y_pred = predictions

# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Definir los nombres de las clases
class_names = ['No Watermark', 'Watermark']

# Crear un dataframe de la matriz de confusión para mostrar los nombres de las clases
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Crear el heatmap de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Calcular True Positives (TPs), True Negatives (TNs), False Positives (FPs), False Negatives (FNs)
TPs = conf_matrix[0, 0]
TNs = conf_matrix[1, 1]
FPs = conf_matrix[1, 0]
FNs = conf_matrix[0, 1]

# Calcular True Positive Rate (TPR), False Positive Rate (FPR), Precision
TPR = TPs / (TPs + FNs)
FPR = FPs / (FPs + TNs)
Precision = TPs / (TPs + FPs)

# Imprimir métricas
print("True Positives (TPs) - not-watermarked:", TPs)
print("True Negatives (TNs) - watermarked:", TNs)
print("False Positives (FPs) - watermarked predicted as not-watermarked:", FPs)
print("False Negatives (FNs) - not-watermarked predicted as watermarked:", FNs)
print("True Positive Rate (TPR) - not-watermarked:", TPR)
print("False Positive Rate (FPR) - watermarked:", FPR)
print("Precision:", Precision)