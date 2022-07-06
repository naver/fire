FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# setup environment
ENV     LANG C.UTF-8
ENV     LC_ALL C.UTF-8
ENV     DEBIAN_FRONTEND noninteractive
# arguments
ARG SOURCE_PREFIX="/opt/src"

RUN mkdir -p ${SOURCE_PREFIX}
# workaroud GPG error
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key del 7fa2af80
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && \
    apt-get install -y sudo ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# Get dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install -c conda-forge --yes \
    numpy \ 
    pyaml \
    faiss-gpu \
    matplotlib \
    tqdm

RUN /opt/conda/bin/pip install kapture

WORKDIR ${SOURCE_PREFIX}
RUN git clone https://github.com/gtolias/how
ENV PYTHONPATH "${PYTHONPATH}:/opt/src/how"

RUN wget "https://github.com/filipradenovic/cnnimageretrieval-pytorch/archive/v1.2.zip" && \
    unzip v1.2.zip && \
    rm v1.2.zip
ENV PYTHONPATH "${PYTHONPATH}:/opt/src/cnnimageretrieval-pytorch-1.2"

RUN git clone https://github.com/jenicek/asmk.git && \
    cd asmk && \
    python3 setup.py build_ext --inplace && \
    rm -r build
ENV PYTHONPATH "${PYTHONPATH}:/opt/src/asmk"

RUN mkdir -p /root/.cache/torch/hub/checkpoints/
WORKDIR /root/.cache/torch/hub/checkpoints/
RUN wget https://download.pytorch.org/models/resnet50-19c8e357.pth

WORKDIR ${SOURCE_PREFIX}
RUN git clone https://github.com/naver/fire.git
# ADD      . ${SOURCE_PREFIX}/fire

### FINALIZE ###################################################################
# save space: purge apt-get
RUN rm -rf /var/lib/apt/lists/*
USER root
WORKDIR ${SOURCE_PREFIX}/fire
