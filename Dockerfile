FROM python:3.6.14-bullseye
WORKDIR /home/tods
COPY . /home/tods
RUN ls
RUN apt-get update && apt-get install -y libssl-dev libcurl4-openssl-dev libyaml-dev build-essential libopenblas-dev libcap-dev ffmpeg
RUN git clone https://github.com/tykimseoul/tods.git
RUN pip3 install -e . --use-deprecated=legacy-resolver