# Clustering de palabras
Agrupamiento de palabras según su relación sintáctica y semántica utilizando el argoritmo de k-means. 

Trabajo perteneciente a la cátedra "Minería de Datos para Texto" de Laura Alonso Alemany - FaMAF UNC. 2017
El corpus ultilizado es un dump de notas periodísticas del diario La Voz del Interior.

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
        - Triplas de dependencia.
    * Eliminación de stopwords de los diccionarios de las palabras.
    * Eliminación de palabras poco frecuentes en el corpus, de los diccionarios de las palabras.
    * Eliminación de palabras poco frecuentes como contexto, de los diccionarios de las palabras.
7. Vectorización de las palabras.
8. Normalización de la matriz (número de ocurrencias totales de la columna sobre ocurrencias por cada fila).

### Clustering
1. Elección de número de clusters.
2. Centroides aleatorios.
3. Algoritmo de k-means usando la distancia coseno para crear los clusters.
2. Iteración a tres valores distintos de k.

## Procedimiento en detalle
Utilizamos la herramienta Scapy para separar en oraciones, en palabras y etiquetar cada palabra con su POS tag, morfología del tag y funcionalidad en la oración.

      nlp = spacy.load('es', vectors=False, entity=False)
      doc = nlp(dataset)
      
Las palabras pueden accederse de dos maneras:
Desde cada oración:

      doc.sents[0]
      
Desde todo el documento:

      doc[0]

Quitamos las oraciones con menos de 10 palabras:

      sents = [sent for sent in doc.sents if len(sent) > 10]

Creamos una lista con las palabras procesadas, evitando aquellas que sean puntuaciones, números o desconocidas.

      words = []
      words_lemma = []
      for sent in sents:
          for word in sent:
              if word.is_alpha:
                  words.append(word)
                  words_lemma.append(word.lemma_)

Contamos las ocurrencias totales de cada palabra en el corpus

      counts = Counter(words_lemma)
      
Usamos un diccionario de lemas, ya que el lematizador que utiliza Scapy en español sólo transforma las palabras en lowercase.
Este diccionario es una herramienta útil aunque no es precisa ya que no considera la función que cumple la palabra en la oración, lo cual genera problemas en palabras que son ambiguas.
      
      lemma_file = open("lemmatization-es.txt", "r")

Recorremos la lista de palabras descartando aquellas que ocurren pocas veces y las agregamos a un diccionario. Si la palabra fue agregada anteriormente, traemos el contexto para modificarlo.

      for word in words:
          w = lemmatize(word.lemma_)
          if not word.is_alpha or str.isdigit(w) or counts[w] < threshold_w:
              continue
          if not w in dicc:
              contexts = {}
          else:
              contexts = dicc[w]

Agregamos los features. 
Primero agregamos su POS tag.

      pos = "POS__" + word.pos_
      if not pos in contexts:
         contexts[pos] = 0
      contexts[pos] += 1
      
Luego agregamos su funcionalidad.

      dep = "DEP__" + word.dep_
      if not dep in contexts:
         contexts[dep] = 0
      contexts[dep] += 1

Luego, la morfología del tag, siendo ésta parseada previamente ya que se encuentran unidas en un solo string

      tags = parse_tags(word)
      for tag in tags:
         if not tag in contexts:
            contexts[tag] = 0
         contexts[tag] += 1
        
Agregamos los contextos (ventana de 1 palabra)

      if not word.i == 0:
           context_izq = doc[word.i - 1]
           c_izq = lemmatize(context_izq.lemma_)
           if context_izq.is_alpha and counts[c_izq] > threshold_c:
               if str.isdigit(c_izq):
                   c_izq = "NUM__"
               if not c_izq in contexts:
                   contexts[c_izq] = 0
               contexts[c_izq] += 1

       if not word.i < len(doc):
           context_der = doc[word.i + 1]
           c_der = lemmatize(context_der.lemma_)
           if context_der.is_alpha and counts[c_der] > threshold_c:
               if str.isdigit(c_der):
                   c_der = "NUM__"
               if not c_der in contexts:
                   contexts[c_der] = 0
               contexts[c_der] += 1

Agregamos la tripla de dependencia: palabra__lemma__funcionalidad__palabra-head-del-arbol-de-dependencia-lematizada

    tripla = "TRIPLA__" + w + "__" + word.lemma_ + "__" + word.dep_ + "__" + lemmatize(word.head.lemma_)
    if not tripla in contexts:
        contexts[tripla] = 0
    contexts[tripla] += 1

Separamos las palabras y sus features

      features = []
      key_words = {}
      wid = 0
      for d in dicc:
          if len(d) > 0:
              key_words[d] = wid
              wid += 1
              features.append(dicc[d])
              
Vectorizamos las palabras con Sklearn

      from sklearn.feature_extraction import DictVectorizer
      v = DictVectorizer(sparse=False)
      matrix = v.fit_transform(features)
      
Normalizamos la matriz

      matrix_normed = matrix / matrix.max(axis=0)
      
Ahora creamos los clusters:
Utilizamos el algoritmo de kmeans de nltk para crear juntar en clusters. Las semillas son aleatorias. Iteramos a 3 valores. 

      clusterer = kmeans.KMeansClusterer(k, cosine_distance, avoid_empty_clusters=True)
      clusters = clusterer.cluster(matrix_normed, True)

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
