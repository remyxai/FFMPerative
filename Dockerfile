FROM jrottenberg/ffmpeg:6.0-ubuntu

RUN apt-get update && apt-get upgrade -y && apt-get install libsndfile1

RUN apt-get install -y python3 python3-pip

RUN pip3 install ffmperative && pip3 install SoundFile

ENTRYPOINT ["/bin/bash"]

