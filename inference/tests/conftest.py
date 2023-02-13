import json

import pandas as pd
import pytest
from mlem.api import save
from sklearn.dummy import DummyRegressor


@pytest.fixture(autouse=True)
def models(tmp_path_factory):
    yield tmp_path_factory.mktemp("./models")


@pytest.fixture(autouse=True)
def model(models, usi):
    model_path = models / f"{usi}_model"
    regressor = DummyRegressor()
    X = pd.DataFrame([[1, 2.1], [4, 5.4]], columns=["bar", "baz"])
    y = pd.Series([3.1, 9.4])
    regressor.fit(X, y)
    save(regressor, model_path)
    yield regressor


@pytest.fixture
def usi():
    yield "foo"


@pytest.fixture
def features_dtypes():
    yield {"bar": "int64", "baz": "float64"}


@pytest.fixture(autouse=True)
def model_performance(models, usi, features_dtypes):
    summary = models / f"{usi}_summary.json"
    content = {
        "features": features_dtypes,
    }
    with summary.open("w+") as f:
        json.dump(content, f)
    yield content


@pytest.fixture(autouse=True)
def patch_env(mocker, models, usi):
    mocker.patch("housing_inference.app.MODELS_PATH", models)
    mocker.patch("housing_inference.app.USI", usi)
