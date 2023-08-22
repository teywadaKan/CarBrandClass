FROM python:3.11.4

WORKDIR /carBrandClass

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /carBrandClass/requirements.txt

COPY ./app /carBrandClass/app
COPY ./model /carBrandClass/model

ENV PYTHONPATH "${PYTHONPATH}:/carBrandClass"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]