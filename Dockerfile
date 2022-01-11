# Base image
FROM python:3.7-slim

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install dvc[gs]
COPY .dvc/ .dvc/
COPY data.dvc/ data.dvc/
RUN dvc pull

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY tests/ tests/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

