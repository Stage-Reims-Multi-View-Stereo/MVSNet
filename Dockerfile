# syntax=docker/dockerfile:1
#
# https://github.com/YoYo000/MVSNet
#
#
# singularity run --nv <docker-url>
#
# Dépendances:
#   - cuda 10.0 (voir table de compatibilité tensorflow)
#   - cudnn 7.0
#   - python 2.7
#   - tensorflow : <2, 1.14 doit être ok (https://github.com/YoYo000/MVSNet/issues/80)
#
#
# Ajout de gdown pour pouvoir télécharger des dataset qui sont hébergés sur google drive
# (de très gros fichiers, impossible de télécharger avec wget à cause du msg de confirmation)
#
# Modifications légères du requirements.txt 
#
# Différence entre les versions de CUDA:
#   - base: seulement libcudart
#   - runtime: + les libs CUDA
#   - devel: + outils , headers, etc (seulement celle ci permet nvcc)
# (https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container)
#
# Pour utiliser les GPU dans docker, utiliser docker run --gpus après avoir installé nvidia-container-toolkit
# Il faudra installer nvidia-toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# Evite des problèmes lors de l'installation de certains packages
ARG DEBIAN_FRONTEND=noninteractive

# Prérequis
RUN apt-get update && apt-get install -y \
  curl python3.8 python3-pip python2.7 python2.7-tk \
  zlib1g-dev libjpeg-dev libglib2.0-0 libsm6 \
  libxrender1 libxext6 \
  python3-dev python3-setuptools locales

#################################
# Locale
# Pouvoir utiliser les accents (utf8)
# Sinon impossible d'écrire par ex. 'é' dans le shell
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     
#################################

#################################
# Python 2 pip
WORKDIR /var
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py && \
  python2.7 get-pip.py && \
  python2.7 -m pip install --upgrade 'pip < 21.0'
#################################

#################################
# Python 3 pip + packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install markupsafe gdown jupyter
RUN python3 -m pip install numpy matplotlib
#################################

#################################
# Prérequis MVSNet
WORKDIR /app
COPY requirements.txt .
RUN python2.7 -m pip install -r requirements.txt
#################################

WORKDIR /app
CMD ["/bin/bash"]
