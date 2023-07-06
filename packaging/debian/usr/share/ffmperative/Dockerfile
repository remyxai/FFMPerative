FROM jrottenberg/ffmpeg:6.0-ubuntu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install libsndfile1 -y && apt-get install libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 git -y

RUN apt-get install -y python3 python3-pip

RUN pip3 install ffmperative && pip install git+https://github.com/m-bain/whisperx.git

RUN pip install huggingface_hub

# Copy entrypoint script into the image
COPY entrypoint.sh /entrypoint.sh

# Set the entry point to our script
ENTRYPOINT ["/entrypoint.sh"]
