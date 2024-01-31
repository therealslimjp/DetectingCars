FROM  python:3.11-slim-bullseye
LABEL authors="jan-philip"


WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

EXPOSE 7860
CMD ["python","-u","app.py"]