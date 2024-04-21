# WatermarkedDetector

<img src="https://javier.rodriguez.org.mx/itesm/2014/tecnologico-de-monterrey-blue.png" width="50%">

## Tecnológico de Monterrey Campus Querétaro
### *Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 201)*
### *Jordana Betancourt Menchaca A01707434*

# Watermarks vs Not-Watermarks

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
#### Papers utilizados en la implementación:
- [Large-Scale Visible Watermark Detection and Removal with Deep Convolutional Networks](https://link.springer.com/chapter/10.1007/978-3-030-03338-5_3)
- [Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
- [Image Classification Based On CNN: A Survey](https://d1wqtxts1xzle7.cloudfront.net/90135273/p2-libre.pdf?1661255798=&response-content-disposition=inline%3B+filename%3DImage_Classification_Based_On_CNN_A_Surv.pdf&Expires=1713721022&Signature=NMCdEZzBOcx8flTi9OSuaUhDZV68yQGmlqnLRsYBB0P~1FoWClcLntffqWDrbKJiGMTxyxErKjDgf~iXbeYVaecMQxeRCWFXHlnAIsgvABD1ZKGXYb2v2c~2-UVe3sJr-t148s~chjp6Cvhvxdn-GXwm6ZnDcbsbqgXsHGOZVhvD0aLvcLX28zJlhIfChzViS1OG1~VGfxdxMrt3nzCKK~MHWP8yOHDuAX40xeFWjcK~HKRe2y2Yt15Ka4C~WzhQFHOOHcfiVbUmVUcejmP~22j~CNrw3T0BZKef9pVfT8BRby~WV3q4pkdBKjaZ~l-~2PmZYfNWAsy7RTycJqL6aA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
- [Binary cross entropy with deep learning technique for
Image classification](https://www.researchgate.net/profile/Vamsidhar-Yendapalli/publication/344854379_Binary_cross_entropy_with_deep_learning_technique_for_Image_classification/links/5f93eed692851c14bce1ac68/Binary-cross-entropy-with-deep-learning-technique-for-Image-classification.pdf)

---

# Preprocesamiento aplicado:
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

## Generadores de datos de imágenes:

Se crean generadores de datos de imágenes para entrenamiento (`train_generator`) y validación (`validation_generator`) utilizando los directorios de datos de entrenamiento y validación correspondientes.
Ambos generadores tienen un tamaño de objetivo de (100, 100) para redimensionar las imágenes a 100x100 píxeles.
Se especifica un tamaño de lote de 32 para ambos generadores.

## Función `safe_image_generator`:

Se define una función `safe_image_generator` para manejar errores comunes durante la generación de lotes de imágenes.
Esta función verifica si una imagen es `None` y, en ese caso, la omite.
También maneja la excepción cuando una imagen está truncada, imprimiendo un mensaje y omitiendo la imagen problemática.
