FROM python:3.10

ARG MODEL_USI
ENV MODEL_USI=${MODEL_USI}
EXPOSE 8080

WORKDIR /app
COPY training/models/${MODEL_USI}_requirements.txt .

ADD inference/ .
RUN python -m pip install -r ${MODEL_USI}_requirements.txt && python -m pip install -r requirements.txt

CMD ["make", "run"]
