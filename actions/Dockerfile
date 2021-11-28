FROM rasa/rasa-sdk:2.8.2

WORKDIR /app

COPY requirements.txt ./

USER root

COPY . /app/actions

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

USER 1000