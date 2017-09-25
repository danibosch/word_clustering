# Clustering de palabras

## Instalación
    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    $ conda create --name keras python=3.5
    $ source activate keras
    (keras) $ conda install --yes --file requirements.txt
    (keras) $ python -m spacy download es
    (keras) $ export tfBinaryURL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl
    (keras) $ pip install $tfBinaryURL
    (keras) $ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
    (keras) $ conda install -c conda-forge keras
    (keras) $ jupyter notebook

    KERAS_BACKEND=tensorflow jupyter notebook
