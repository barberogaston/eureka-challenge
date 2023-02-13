FROM jupyter/datascience-notebook

USER root
WORKDIR /app
RUN mkdir /mlflow && \
    chmod 777 /mlflow && \
    python -m pip install --pre pycaret && python -m pip install pandas_profiling xgboost optuna mlflow mlem
