import os

from fastapi import FastAPI, Request

from housing_inference.api.load import load_model, load_performance
from housing_inference.api.parse import parse_params
from housing_inference.api.predict import predict


USI = os.getenv("MODEL_USI")
MODELS_PATH = os.getenv("MODELS_PATH")
app = FastAPI()

@app.on_event("startup")
def load_caches():
    load_model(MODELS_PATH, USI)
    load_performance(MODELS_PATH, USI)


@app.get("/model_performance")
async def get_model_performance():
    """
    Get the model's training metrics as well as some other metadata.
    """
    return load_performance(MODELS_PATH, USI)


@app.get("/predict")
async def get_predict(request: Request):
    """
    Get a model's predictions by passing features as query parameters.
    """
    model = load_model(MODELS_PATH, USI)
    features_dtypes = load_performance(MODELS_PATH, USI)["features"]
    data = parse_params(request, features_dtypes)
    return predict(model, data)
