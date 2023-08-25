FROM python:3.11.4

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN /usr/local/bin/python -m pip install --upgrade pip &&\
    pip install -r /app/requirements.txt &&\
    mkdir /app/config &&\
    mkdir /app/dataset

COPY ./*.py /app/
COPY ./model_util/*.py /app/model_util/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]
