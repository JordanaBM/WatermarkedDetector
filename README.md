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

**Preprocesamiento aplicado:** Se aplicó un preprocesamiento básico que incluye redimensionamiento a 100x100 píxeles y escalamiento de los valores de píxeles para que estén entre 0 y 1.

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
