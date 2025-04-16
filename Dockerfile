
# FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
# FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get -y install \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    xvfb \
    build-essential \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    wget \
    unzip \
    software-properties-common


RUN ln -s /usr/bin/python3 /usr/bin/python

FROM python:3.6

RUN pip install --upgrade pip
RUN pip install \
    "tensorflow>=1.8.0,<2.0" \
    gym[all] \
    joblib \
    cloudpickle \
    matplotlib \
    ipython \
    mujoco-py==2.1.2.14  


RUN git clone https://github.com/openai/spinningup.git /spinningup
WORKDIR /spinningup
RUN pip install -e . --no-deps



ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin


#COPY entrypoint.sh /usr/local/bin/
#RUN chmod +x /usr/local/bin/entrypoint.sh
#ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

