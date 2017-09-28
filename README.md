# Clustering de palabras

## Procedimiento
### Procesamiento del corpus
1. Separación del corpus en oraciones y en palabras.
2. Análisis morfosintáctico y funcionalidad de cada palabra.
3. Eliminación de oraciones con menos de 5 palabras.
4. Conteo de ocurrencias totales de cada palabra.
5. Creación de diccionario de palabras.
5.1. Agregado de palabras al diccionario, aquellas que no sean números, puntuaciones o desconocidas. Cada palabra contiene un diccionario.
5.2. Agregado al diccionario de cada palabra:
5.2.1. Palabras de contexto. Aquellas que ocurren en la misma oración, sin orden.
5.2.2. Part-of-speech tag.
5.2.3. Morfología de tag.
5.2.4. Funcionalidad.
5.2.5. Triplas de dependencia.
5.3. Eliminación de stopwords de los diccionarios de las palabras.
5.4. Eliminación de palabras poco frecuentes en el corpus de los diccionarios de las palabras.
6. Vectorización de las palabras.
7. Normalización de la matriz.

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
