FROM jrottenberg/ffmpeg:6.0-ubuntu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && \
    apt-get install libsndfile1 libass-dev libsm6 \
    libxext6 libgl1-mesa-glx libglib2.0-0 \
    git python3 python3-pip -y

RUN pip3 install git+https://github.com/m-bain/whisperx.git ffmperative[full] huggingface_hub

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
