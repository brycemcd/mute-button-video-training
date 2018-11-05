# docker build -t football_sample_and_model .
FROM python:3.6

ENV APP_DIR /opt/project

WORKDIR $APP_DIR

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
