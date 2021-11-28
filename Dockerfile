FROM rasa/rasa:2.8.14-full
WORKDIR /app
COPY . .
USER root
RUN pip install vader-multi