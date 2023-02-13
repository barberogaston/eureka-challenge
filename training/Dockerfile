FROM jupyter/datascience-notebook

USER root
WORKDIR /app
COPY requirements.txt .
RUN mkdir /mlflow && \
    chmod 777 /mlflow && \
    python -m pip install -r requirements.txt
