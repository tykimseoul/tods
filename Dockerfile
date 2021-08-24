FROM python:3.6.14-bullseye
RUN apt-get update && apt-get install -y libssl-dev libcurl4-openssl-dev libyaml-dev build-essential libopenblas-dev libcap-dev ffmpeg
WORKDIR /home/tods
RUN git clone https://github.com/tykimseoul/tods.git
WORKDIR ./tods
RUN pip3 install -e . --use-deprecated=legacy-resolver