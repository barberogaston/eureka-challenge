# Housing Prices Challenge

## Requirements

### 🐳 Docker

I used Docker Engine `v20.10.21` and Docker Compose `v2.12.2`, so anything above those shall work fine.

### 🐍 Python

Although I suggest running everything using Docker, if you wish to run things locally, prefer using Python 3.10 onwards.

### 💻 Platform

This repository was developed on a M1 MacBook Pro with MacOS `v12.6.3`.

### 🥲 Misc

As we all know, most ML libraries have either C or Fortran code (or both). In order to compile these libraries, other native system libraries may be required. This is heavily dependant on the user's Operating System and CPU architecture, which makes it almost impossible to cover all edge cases. The best recommendation for this is to try to stick with running the docker containers and (if possible) on a x86_64 Linux system.

## 🧩 Services

### 🧮 Experiments

The [experiments](./experiments) folder just holds a [Dockerfile](./experiments/Dockerfile) and some other files which MLFlow uses to keep track of every trained model with the [Training](#-training) service.

#### Execution

```bash
docker compose build mlflow && \
docker compose up mlflow
```

#### Dashboard

Go to [http://127.0.0.1:5000](http://127.0.0.1:5000) once the container has started.

### 🎛 Training

This service lives under the [training](./training) folder. In here, you'll find all the necessary code and assets to train the machine learning model for the housing prices challenge. Once inside JupyerLab, open the [Training.ipynb](./training/Training.ipynb) notebook. It should be executed as-is.

The [models](./training/models/) folder is where every trained and fine tuned model, as well as other meta-files, are saved after the complete execution of the notebook.

#### Execution

_From the root folder of the project:_

```bash
docker compose build training mlflow && \
docker compose up training mlflow
```

Once created, search the terminal logs for Jupyer Lab's url with the generated token.

### 🎯 Inference

All code under the [inference](./inference) folder is used to serve a model generated with the [Training](#-training) service.

#### Execution

##### 🏠 Local

> ⚠️ Remember to activate a virtual environment

_From the [inference](./inference/) folder do:_

```bash
make install && \
make run
```

##### 🐳 Container

_From the root folder of the project do:_

```bash
docker compose build inference && \
docker compose up inference
```

#### Serving another model

In order to server another model than the one that comes already pre-trained, you must modify the [docker-compose.yml](docker-compose.yml) file, where it says `MODEL_USI=65bc` and put the four characters at the beggining of your model's assets, inside the [training/models](./training/models/) folder, rebuild the inference container and start it again.

#### API

`GET /model_performance`

##### Request

```bash
curl http://127.0.0.1:8080/model_performance
```

##### Response

```json
{
    "features": {
        "property_area": "int64",
        "house_age": "int64",
        "house_style": "object",
        "neighborhood": "object",
        "overall_quality": "int64",
        "overall_condition": "int64",
        "spaciousness": "int64",
        "remodel_age": "int64",
        "bath_area": "float64",
        "bsmt_area": "int64",
        "garage_area": "int64",
        "garage_age": "float64",
        "has_2ndfloor": "int64",
        "has_porch": "int64",
        "has_multiple_kitchen": "int64"
    },
    "artifact": {
        "model": "lightgbm",
        "usi": "65bc"
    },
    "cv": {
        "folds": 10,
        "random_state": 42,
        "strategy": "KFold"
    },
    "train": {
        "r2": {
            "mean": 0.7764825715930765,
            "std": 0.0007612262956481716
        },
        "mae": {
            "mean": 24092.349484500835,
            "std": 31.70313569401462
        },
        "mse": {
            "mean": 1042029523.8216851,
            "std": 3151468.276935437
        },
        "rmse": {
            "mean": 32280.445136102855,
            "std": 48.842980445993916
        }
    },
    "test": {
        "r2": {
            "mean": 0.7555913883519633,
            "std": 0.007640956396903779
        },
        "mae": {
            "mean": 25047.75358732067,
            "std": 264.0812641543778
        },
        "mse": {
            "mean": 1138422022.3600926,
            "std": 32336522.40491208
        },
        "rmse": {
            "mean": 33737.12125366556,
            "std": 478.19648217022956
        }
    }
}
```

`GET /predict`

##### Request

We'll use python since passing many query parameters in `curl` is quite hard.

```python
import requests

params = {
    "property_area": 2501,
    "house_age": 12,
    "house_style": "2Story",
    "neighborhood": "Gilbert",
    "overall_quality": 6,
    "overall_condition": 5,
    "spaciousness": 207,
    "remodel_age": 15,
    "bath_area": 2.5,
    "bsmt_area": 775,
    "garage_area": 386,
    "garage_age": 12,
    "has_2ndfloor": 1,
    "has_porch": 1,
    "has_multiple_kitchen": 0,
}

response = requests.get("http://127.0.0.1:8080/predict", params=params)

print(response.status_code)
print(response.json())
```

##### Output

```python
200
{'prediction': 159198.1764862322}
```

#### Tests

For running tests you can opt for doing it locally or by executing the image and running them inside the container.

##### 🏠 Local

> ⚠️ Remember to activate a virtual environment

_From the [inference](./inference/) folder do:_

```bash
make install && \
make test
```

##### 🐳 Container

_From the root folder of the project do:_

```bash
docker compose build inference && \
docker compose run inference make test
```
