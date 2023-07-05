FROM jrottenberg/ffmpeg:6.0-ubuntu

RUN apt-get update && apt-get upgrade -y && apt-get install libsndfile1 -y && apt-get install libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 -y

RUN apt-get install -y python3 python3-pip

RUN pip3 install ffmperative

ENTRYPOINT ["/bin/bash"]

