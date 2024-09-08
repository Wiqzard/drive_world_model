FROM nvcr.io/nvidia/mxnet:24.02-py3 
#nvcr.io/nvidia/k8s/container-toolkit:v1.16.1-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9 and pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Install xformers
RUN MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="9.0" python -m pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers


