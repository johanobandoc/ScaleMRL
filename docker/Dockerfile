##################################################
# Build instructions:                            #
# docker build . -t <user>/newt:1.0.0            #
# docker push <user>/newt:1.0.0                  #
##################################################

# base image
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

# packages
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends build-essential git nano rsync vim tree curl wget \
    swig ffmpeg unzip htop tmux xvfb ca-certificates bash-completion libjpeg-dev libpng-dev \
    libssl-dev libcurl4-openssl-dev libopenmpi-dev zlib1g-dev qtbase5-dev qtdeclarative5-dev \
    libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 libgl1-mesa-glx libosmesa6 \
    libosmesa6-dev libglew-dev mesa-utils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# conda environment
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
COPY environment.yaml /root
RUN conda update conda && \
    conda env update -n base -f /root/environment.yaml && \
    rm /root/environment.yaml && \
    conda clean -ya && \
    pip cache purge && \
    conda init
SHELL ["/bin/bash", "-c"]
RUN echo "cd /root" >> /root/.bashrc

# pip packages
RUN pip install --no-cache-dir 'ale_py==0.10'

# download physx GPU binary via sapien
RUN python -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"

# copy maniskill assets to avoid repeatedly downloading
COPY .maniskill /root/.maniskill

# misc environment variables for convenience
ENV TORCH_DISTRIBUTED_TIMEOUT=1800
ENV MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

# success!
RUN echo "Successfully built Newt Docker image!"
