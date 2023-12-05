FROM smellslikeml/ffmperative-base:0.0.6

ARG DEBIAN_FRONTEND=noninteractive

RUN git clone https://github.com/remyxai/FFMPerative.git && \
    cd FFMPerative && \
    git checkout minimal_pkg && \
    pip3 install requests && \
    pip3 install -r requirements.txt && \
    python3 setup.py install

WORKDIR /

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
