FROM jrottenberg/ffmpeg:6.0-ubuntu

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y python3 python3-pip

RUN pip3 install ffmperative

ENTRYPOINT ["/bin/bash"]

