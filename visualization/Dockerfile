FROM  python:3.11-slim-bullseye
LABEL authors="jan-philip"



WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app

EXPOSE 7860
ENTRYPOINT ["python3", "app.py"]