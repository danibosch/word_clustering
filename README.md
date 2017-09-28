# Clustering de palabras

## Procedimiento
### Procesamiento del corpus
1. Separación del corpus en oraciones y en palabras.
2. Análisis morfosintáctico y funcionalidad de cada palabra.
3. Eliminación de oraciones con menos de 5 palabras.
4. Lematización de palabras.
5. Conteo de ocurrencias totales de cada palabra.
6. Creación de diccionario de palabras.
    * Agregado de palabras al diccionario, aquellas que no sean números, puntuaciones o desconocidas. Cada palabra contiene un diccionario.
    * Agregado al diccionario de cada palabra:
        - Palabras de contexto. Aquellas que ocurren en la misma oración, sin orden.
        - Part-of-speech tag.
        - Morfología de tag.
        - Funcionalidad.
        - Triplas de dependencia (palabra__funcionalidad__palabra-head-del-arbol-de-dependencia-lematizada).
    * Eliminación de stopwords de los diccionarios de las palabras.
    * Eliminación de palabras poco frecuentes en el corpus, de los diccionarios de las palabras.
    * Eliminación de palabras poco frecuentes como contexto, de los diccionarios de las palabras.
7. Vectorización de las palabras.
8. Normalización de la matriz (número de ocurrencias totales de la columna sobre ocurrencias por cada fila).

## Clustering
1. Elección de número de clusters.
2. Centroides aleatorios.
3. Algoritmo de k-means usando la distancia coseno para crear los clusters.
2. Iteración a tres valores distintos de k.

## Resultados
k = 50
k = 100
k = 150

## Instalación
    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    $ conda create --name keras python=3.5
    $ source activate keras
    (keras) $ conda install --yes --file requirements.txt
    (keras) $ python -m spacy download es
    (keras) $ export tfBinaryURL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl
    (keras) $ pip install $tfBinaryURL
    (keras) $ conda install -c conda-forge keras
    (keras) $ jupyter notebook

    KERAS_BACKEND=tensorflow jupyter notebook
