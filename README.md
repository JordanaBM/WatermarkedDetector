### - [modelo_cnn_watermarks1.h5](https://drive.google.com/file/d/1id7tUkzsG-WnulKZBM14kHdY9QLOPuI3/view?usp=sharing) No me dejó subirlo a Github porque pesa 76 mb.

# WatermarkedDetector

<img src="https://javier.rodriguez.org.mx/itesm/2014/tecnologico-de-monterrey-blue.png" width="50%">

## Tecnológico de Monterrey Campus Querétaro
### *Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 201)*
### *Jordana Betancourt Menchaca A01707434*

# 1. Watermarks vs Not-Watermarks Dataset

**Descripción breve:** Este dataset contiene imágenes con y sin marcas de agua, recopiladas a partir del dataset "Watermarked / Not watermarked images" de Kaggle, el cual se basa en imágenes gratuitas obtenidas de Pexels.com. Las imágenes están etiquetadas adecuadamente y se pueden utilizar para tareas de clasificación de imágenes.

**Fuente de los datos:** Los datos obtenidos provienen del dataset titulado [Watermarked / Not watermarked images](https://www.kaggle.com/datasets/felicepollano/watermarked-not-watermarked-images) de Kaggle, que a su vez se obtienen a partir de imágenes gratuitas obtenidas de Pexels.com.

**Fecha de creación o actualización:** El dataset fue creado en 2020 y actualizado por mí el 12 de abril de 2024.

**Autores o creadores:** Fellice Pollano, Jordana Betancourt Menchaca.

**Formato de los datos:** Imágenes.

**Tamaño del dataset:**  31, 575 imágenes.

- Watermark: 10,008 train + 2,502 validation  + 3,299 test
- Sin watermark: 9,982 train + 2,495 validation  + 3,289 test

**Variables o características:** El dataset contiene imágenes con y sin marcas de agua, etiquetadas adecuadamente para su uso en tareas de clasificación.

**Valores faltantes:** Ninguno.

**Distribución de las clases:**
- Watermark: 15,809 imágenes (50.06%)
- -- Train (63.30%) + Validation (15.82%) + Test (20.86%)
- Sin watermark: 15,766 imágenes (49.94%)
- -- Train (63.31%) + Validation (15.82%) + Test (20.86%)

**Licencia:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) -- Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional.

- Atribución (BY): Se permite copiar, distribuir, exhibir y ejecutar el trabajo, así como hacer y distribuir trabajos derivados, siempre que se reconozca y se dé crédito al autor original de la manera especificada por éste.
- NoComercial (NC): Se permite copiar, distribuir, exhibir y ejecutar el trabajo, así como hacer y distribuir trabajos derivados, pero no para un fin comercial sin permiso previo del autor.
- CompartirIgual (SA): Se permite hacer y distribuir trabajos derivados, pero solo bajo una licencia idéntica a la que regula el trabajo original (es decir, licencia CC BY-NC-SA 4.0).

**URL de descarga:** [Enlace de descarga](https://drive.google.com/drive/folders/1DMZaVaJ7cLMuh9MQibmfjir5RspuMbVq?usp=sharing).

**Ejemplos de uso:** Estos datos pueden ser utilizados para entrenar modelos de aprendizaje automático para detectar y eliminar marcas de agua en imágenes.

**Agradecimientos:** Agradezco a Pexels por proveer imágenes de alta calidad de manera gratuita, y a Fellice por realizar un dataset que ya se encontraba en buen estado para utilizar.

**Referencias:**
- [Watermark Removal using Convolutional Autoencoder](https://www.kaggle.com/code/ankit8467/watermark-removal-using-convolutional-autoencoder)
- [Cat Eyes](https://www.kaggle.com/code/mpwolke/cat-eyes)
## Papers utilizados en la implementación:
- [Large-Scale Visible Watermark Detection and Removal with Deep Convolutional Networks](https://link.springer.com/chapter/10.1007/978-3-030-03338-5_3)
- [Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
- [Image Classification Based On CNN: A Survey](https://www.researchgate.net/publication/355800126_Image_Classification_Based_On_CNN_A_Survey)
- [Binary cross entropy with deep learning technique for
Image classification](https://www.researchgate.net/profile/Vamsidhar-Yendapalli/publication/344854379_Binary_cross_entropy_with_deep_learning_technique_for_Image_classification/links/5f93eed692851c14bce1ac68/Binary-cross-entropy-with-deep-learning-technique-for-Image-classification.pdf)

---

# 2. Preprocesamiento aplicado
## ImageDataGenerator para entrenamiento y validación:

Se crean dos instancias de `ImageDataGenerator`, una para datos de entrenamiento (`train_datagen`) y otra para datos de validación (`validation_datagen`).
Ambas instancias aplican las siguientes transformaciones de aumento de datos a las imágenes:
- `rescale=1./255`: Normaliza los valores de píxeles al rango [0, 1] dividiendo cada valor de píxel por 255.
- `rotation_range=20`: Rango de rotación aleatoria de hasta 20 grados.
- `zoom_range=0.2`: Rango de zoom aleatorio.
- `width_shift_range=0.2`: Rango de desplazamiento horizontal aleatorio.
- `height_shift_range=0.2`: Rango de desplazamiento vertical aleatorio.
- `horizontal_flip=True`: Volteo horizontal aleatorio.
- `vertical_flip=True`: Volteo vertical aleatorio.

**Justificación:** Las transformaciones aplicadas por `ImageDataGenerator` introducen una variedad en los datos de entrenamiento y validación, lo que ayuda al modelo a ser más robusto y a generalizar mejor. La normalización de los valores de píxeles asegura que los datos estén en un rango adecuado para el modelo, mientras que las transformaciones de rotación, zoom, desplazamiento y volteo introducen variabilidad que puede mejorar la capacidad del modelo para reconocer distintas variaciones de los watermarks.


## Generadores de datos de imágenes:

Se crean generadores de datos de imágenes para entrenamiento (`train_generator`) y validación (`validation_generator`) utilizando los directorios de datos de entrenamiento y validación correspondientes.
Ambos generadores tienen un tamaño de objetivo de (100, 100) para redimensionar las imágenes a 100x100 píxeles.
Se especifica un tamaño de lote de 32 para ambos generadores.

**Justificación:** Redimensionar las imágenes a una dimensión común facilita el procesamiento para el modelo.

## Función `preprocess_images`:

Se ha creado una función llamada `preprocess_images` que se encarga de procesar imágenes dentro de un directorio específico. Su función principal consiste en eliminar aquellas imágenes que no se pueden abrir correctamente, ya sea por archivos dañados, formatos incorrectos u otros problemas similares.

**Justificación:** Es útil para limpiar un directorio de imágenes, eliminando aquellas que están corruptas o en un formato no compatible. Esto es importante en tareas de preprocesamiento de datos para modelos de aprendizaje automático, donde tener datos de entrada limpios y válidos es fundamental para el entrenamiento correcto del modelo. Eliminar las imágenes no válidas ayuda a evitar errores durante el procesamiento y el entrenamiento del modelo, lo que puede conducir a resultados más confiables y precisos.

---

# 3. Selección del algoritmo

## Detector de Watermarks como problema de detección de objetos

Un detector de watermarks puede considerarse un problema de detección de objetos porque implica identificar y localizar la presencia de un objeto específico, en este caso, el watermark, dentro de una imagen más grande. Aunque tradicionalmente se ha pensado en la detección de objetos en imágenes como la identificación de objetos físicos como personas, autos o animales, en este contexto, el watermark puede considerarse un tipo especial de objeto que se busca identificar y ubicar en una imagen.

La detección de watermarks como un problema de detección de objetos implica adaptar y aplicar técnicas de detección de objetos existentes para identificar la presencia y la ubicación de watermarks dentro de una imagen. Esto puede requerir el uso de algoritmos y modelos que puedan aprender a reconocer las características únicas de un watermark, como su forma, tamaño, posición relativa y otros atributos visuales, para poder detectarlo de manera automática y precisa en diferentes contextos y condiciones.

La elección entre utilizar una Convolutional Neural Network (CNN) o modelos como RetinaNet o YOLO (You Only Look Once) para la detección de watermarks depende de varios factores, incluyendo la complejidad de los patrones de watermark a detectar, la cantidad de datos disponibles y los requisitos de velocidad y eficiencia del modelo.

1. **Complejidad de los patrones de watermark**: Si los patrones de watermark son relativamente simples y bien definidos, una CNN puede ser suficiente para detectarlos con precisión. Sin embargo, si los patrones son más complejos y variados, modelos como RetinaNet o YOLO, que están diseñados para detectar múltiples objetos en una sola imagen y pueden capturar mejor la complejidad de los patrones, pueden ser más adecuados.

2. **Cantidad de datos disponibles**: Las CNN suelen requerir grandes cantidades de datos de entrenamiento para aprender patrones complejos. Si se dispone de un conjunto de datos pequeño, una CNN puede no ser capaz de generalizar bien. En este caso, modelos como RetinaNet o YOLO, que pueden ser más eficientes en la utilización de datos de entrenamiento limitados, pueden ser preferibles.

3. **Velocidad y eficiencia del modelo**: RetinaNet y YOLO son conocidos por su eficiencia en la detección de objetos en tiempo real, lo que los hace ideales para aplicaciones que requieren baja latencia, como sistemas de vigilancia o vehículos autónomos. Si la detección de watermarks necesita ser rápida y en tiempo real, estos modelos pueden ser más adecuados que una CNN, que puede ser más lenta debido a su naturaleza más profunda y compleja.

La elección de una Convolutional Neural Network (CNN) para la detección de watermarks se justifica con el hecho de que los patrones de las imágenes del dataset son sencillas, además se cuenta con una buena cantidad de imágenes (31, 575) para que el modelo pueda aprender de manera efectiva.

---

# 4. Configuración del algoritmo

### ¿Cómo es la arquitectura de una CNN?

Las CNN están compuestas por tres tipos de capas: la capa de convolución, la capa de agrupación y la capa completamente conectada (relu y salida). Estas capas se apilan en las CNN, como se muestra en la figura.

![Arquitectura CNN](https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/4ea67ae3-d24d-4f06-8fed-127991864c11)

### Diseño del primer modelo `watermarkscnn.py`

1. **Capa Convolucional (Conv2D):**
   - Se define una capa convolucional con 32 filtros de tamaño 3x3 y función de activación ReLU.
   - La capa espera entradas de imágenes con tamaño 100x100 píxeles y 3 canales de color (RGB).
   - Esta capa aplica un conjunto de filtros a la imagen de entrada. Cada filtro convolucional recorre la imagen y realiza una operación de convolución, que consiste en multiplicar los valores de los píxeles de la imagen por los pesos del filtro y sumar los resultados. Esto ayuda a detectar características como bordes y texturas en la imagen.

2. **Capa de Pooling (MaxPooling2D):**
   - Se aplica una capa de pooling para reducir las dimensiones de la salida de la capa convolucional anterior.
   - En este caso, se utiliza MaxPooling2D con un tamaño de ventana de 2x2.
   - La capa de pooling reduce la dimensionalidad de la imagen al agrupar regiones adyacentes de píxeles y tomar el valor máximo de cada región. Esto ayuda a reducir el tamaño de la representación de la imagen y a extraer características importantes de manera más robusta.
  
3. **Segunda Capa Convolucional (Conv2D):**
   - Se define una segunda capa convolucional con 64 filtros de tamaño 3x3 y función de activación ReLU.
   - Esta capa aplica un conjunto de filtros a la imagen de entrada. Cada filtro convolucional recorre la imagen y realiza una operación de convolución, que consiste en multiplicar los valores de los píxeles de la imagen por los pesos del filtro y sumar los resultados. Esto ayuda a detectar características como bordes y texturas en la imagen.
  
4. **Segunda Capa de Pooling (MaxPooling2D)**
   - Se aplica una segunda capa de pooling para reducir las dimensiones de la salida de la segunda capa convolucional.
   - Nuevamente, se utiliza MaxPooling2D con un tamaño de ventana de 2x2.
   - La capa de pooling reduce la dimensionalidad de la imagen al agrupar regiones adyacentes de píxeles y tomar el valor máximo de cada región. Esto ayuda a reducir el tamaño de la representación de la imagen y a extraer características importantes de manera más robusta.

5. **Tercera Capa Convolucional (Conv2D)**
   - Se define una tercera capa convolucional con 128 filtros de tamaño 3x3 y función de activación ReLU.
   - Esta capa aplica un conjunto de filtros a la imagen de entrada. Cada filtro convolucional recorre la imagen y realiza una operación de convolución, que consiste en multiplicar los valores de los píxeles de la imagen por los pesos del filtro y sumar los resultados. Esto ayuda a detectar características como bordes y texturas en la imagen.
  
6. **Tercera Capa de Pooling (MaxPooling2D)**
   - Se aplica una tercera capa de pooling para reducir las dimensiones de la salida de la tercera capa convolucional.
   - Se utiliza MaxPooling2D con un tamaño de ventana de 2x2.
   - La capa de pooling reduce la dimensionalidad de la imagen al agrupar regiones adyacentes de píxeles y tomar el valor máximo de cada región. Esto ayuda a reducir el tamaño de la representación de la imagen y a extraer características importantes de manera más robusta.

7. **Aplanar la Salida (Flatten):**
   - Esta capa convierte la salida de las capas convolucionales y de pooling en un vector unidimensional. Esto es necesario para pasar los datos a las capas densas que siguen, ya que estas requieren una entrada en formato de vector.

8. **Capa Densa (Dense):**
   - Se define primero una capa densa con 512 neuronas y función de activación ReLU
   - Las capas densas son capas completamente conectadas en las que cada neurona está conectada a todas las neuronas de la capa anterior. Estas capas realizan operaciones lineales y no lineales en los datos, lo que ayuda a aprender representaciones más complejas y abstractas de las características de la imagen.

9. **Capa de Salida (Dense):**
   - La capa de salida tiene una sola neurona con una función de activación sigmoide. Esta configuración es típica para problemas de clasificación binaria, donde la neurona de salida representa la probabilidad de que la imagen contenga o no un watermark. La función sigmoide comprime la salida a un rango entre 0 y 1, que se interpreta como la probabilidad de pertenecer a la clase positiva (con watermark).


---

# 5. Resultados del algoritmo

En la detección de watermarks, generalmente se trata de un problema de clasificación binaria, donde se busca determinar si una imagen contiene o no un watermark. La función `binary_crossentropy` es adecuada para este tipo de problemas, ya que mide la discrepancia entre las probabilidades predichas por el modelo y las etiquetas reales, siendo más efectiva en problemas de clasificación binaria que otras funciones de pérdida como la `categorical_crossentropy`.

La `binary_crossentropy` es una función de pérdida comúnmente utilizada en problemas de clasificación binaria en redes neuronales. Esta función de pérdida mide la diferencia entre las distribuciones de probabilidad predichas por el modelo y las distribuciones reales de los datos.

## Training

### Accuracy 59%
<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/fa5267ab-156b-4d3e-857a-596da8f6733a" width="50%">

### Loss 65%
<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/2a466661-cc3f-4443-95b7-2ae2092c13b4" width="50%">


## Testing

### Matriz de confusión

<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/77541790-e127-4a8c-8628-31cd6d659eb0" width="50%">

### Métricas
<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/7aeddada-a8ab-40ff-b8b8-f40f84a15795" width="50%">

**El modelo de detección de watermarks logró identificar correctamente 129 imágenes sin watermark (True Positives) y 40 imágenes con watermark (True Negatives), lo que representa una tasa de acierto del 81.13% y del 69.92%, respectivamente. Sin embargo, también se observaron 93 casos donde el modelo clasificó incorrectamente imágenes con watermark como si no lo tuvieran (False Positives), y 30 casos donde se predijo incorrectamente la presencia de watermark en imágenes que no lo tenían (False Negatives). Esto resultó en una precisión del 58.11%, que indica la proporción de imágenes correctamente identificadas como sin watermark con respecto a todas las imágenes identificadas como tal por el modelo.**


### Mejora en el umbral de predicción manual

Durante la impresión del número de predicción, se constató que los valores se encontraban notablemente dispersos hacia el lado izquierdo. En consecuencia, la media no alcanzaba el valor de 0.5, y se observó que muchas imágenes con marca de agua tenían una predicción inferior a este umbral. Por ende, se determinó que cualquier predicción superior a 0.40 indicaría la presencia de marca de agua. Tras efectuar este ajuste manual, se obtuvieron los siguientes resultados.

<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/531a1dd5-d21d-421f-8fab-c2f34421febf" width="50%">

<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/189c6ef8-60ba-46ce-8b08-1e0ba8372997" width="50%">

**Con este simple cambio en el test, la accuracy pasó a ser del 69%**
---

# 6. Cambios al algoritmo

1. **Mayor complejidad del modelo:** El nuevo modelo contiene una red convolucional más profunda y compleja, con capas convolucionales adicionales (64, 128, 256, 512 filtros) lo que puede permitir al modelo aprender características más abstractas y complejas de las imágenes.

2. **Uso de optimizador Adam:** Aunque el primer modelo ya contenía el optimizador de Adam, este por defecto, tiene un learning rate de 0.001, en el modelo optimizado contamos ahora con una tasa de aprendizaje reducida de 0.0005, que suele ser más efectiva para problemas de aprendizaje profundo y puede ayudar al modelo a converger más rápido y obtener mejores resultados.

3. **Cambio en el procesamiento de imágenes:** Se eliminaron las funciones de zoom, así como los rangos de desplazamiento de ancho y alto (width y height shift range), debido a que al aplicar zoom a algunas imágenes se eliminaba la marca de agua, y al modificar el ancho y alto con rotaciones, en ocasiones la imagen quedaba deformada, lo que dificultaba percibir la marca de agua.

4. **Aumento de número de épocas** Se aumentó a 20 epoch, esto con la finalidad de tener una mayor convergencia, haciendo que tenga una precisión más alta porque posee más tiempo para aprender patrones más compleja. Además, un aumento en el número de épocas puede actuar como una forma de regularización, ayudando al modelo a generalizar mejor a datos nuevos y no vistos. 

**Justificación:** La incorporación de capas convolucionales adicionales en el modelo de detección de watermarks resulta fundamental para capturar características más complejas y específicas de las imágenes, como los patrones de textura, formas y detalles finos característicos de los watermarks. Este enfoque permite que el modelo pueda identificar con mayor precisión la presencia de watermarks en las imágenes. Asimismo, un modelo más profundo con más capas convolucionales tiene la capacidad de generalizar mejor, lo que implica que puede adaptarse de manera más efectiva a nuevas imágenes y escenarios. Esta capacidad de generalización se refleja en un mejor rendimiento en datos de prueba no vistos, lo que evidencia la eficacia y robustez del modelo en la detección de watermarks en diversas situaciones y condiciones, por otra parte, utilizar una tasa de aprendizaje reducida de 0.0005 en el optimizador Adam en lugar del valor por defecto de 0.001 mejora la eficacia del modelo en problemas de aprendizaje profundo. Esta tasa más baja permite ajustar los pesos de la red de manera más precisa y gradual, evitando que el modelo "salte" sobre mínimos locales y permitiendo una convergencia más rápida hacia una solución óptima. 


---

# 7. Resultados del segundo algoritmo

### Accuracy 76%
<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/25c63ced-d0a6-456d-8b2d-d1822e8b2d6f" width="50%">

### Loss 47%
<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/20a666a6-e8d5-4602-84a0-5e8f55c98bb8" width="50%">


## Testing

### Matriz de confusión

<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/b91b5b11-fc38-41c1-ac50-9932e570cf6f" width="50%">


### Métricas
<img src="https://github.com/JordanaBM/WatermarkedDetector/assets/69861226/f8326eb8-5305-41c8-b162-886b32708843" width="50%">

Como se puede observar, incluso cuando la predicción es mayor a 0.5 para imágenes que sí contienen marcas de agua, el modelo mejorado en la fase de prueba obtiene un puntaje de precisión mayor, siendo solo un 5% menor de lo esperado en la fase de entrenamiento.


---

# 8. Comparación de algoritmos

Ambos códigos implementan modelos de redes neuronales convolucionales (CNN) para la clasificación binaria de imágenes, pero difieren en la complejidad y configuración de las capas.

El modelo optimizado utiliza una arquitectura de red más profunda y compleja, lo que le permite aprender representaciones más abstractas y complejas de las imágenes. Las capas convolucionales con un mayor número de filtros (64, 128, 256 y 512) pueden detectar características más sofisticadas en las imágenes, lo que puede ser crucial para identificar marcas de agua sutiles o complejas. En contraste, el modelo original contaba con solo 3 capas convolucionales de 32, 64 y 128 filtros, lo que limitaba su capacidad para capturar detalles finos.

Ambos modelos mantienen una capa densa con 512 neuronas antes de la capa de salida. Sin embargo, en el modelo optimizado se redujo la tasa de aprendizaje del optimizador Adam, lo que ayudó a mejorar la convergencia del modelo y a evitar que se atasque en mínimos locales durante el entrenamiento.

El entrenamiento del modelo durante 20 épocas en lugar de 10 permitió que el modelo optimizado tuviera suficiente tiempo para aprender patrones complejos en los datos, lo que se reflejó en una mejor precisión en la detección de marcas de agua (76% vs 60%).

En resumen, el modelo mejorado para la detección de marcas de agua utiliza una arquitectura más profunda y compleja, lo que le permite aprender representaciones más sofisticadas de las imágenes y detectar patrones más complejos. Por otro lado, el modelo original era más simple y básico, lo que limitaba su capacidad para la detección de marcas de agua.

