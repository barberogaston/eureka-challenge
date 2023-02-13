from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi import HTTPException

from housing_inference.api.predict import predict


def test_predict(model):
    result = predict(model, pd.DataFrame([[1, 2.9]], columns=["bar", "baz"]))
    assert "prediction" in result
    assert isinstance(result["prediction"], float)


def test_predict_failure():
    model = Mock()
    model.predict = Mock(side_effect=Exception)

    with pytest.raises(HTTPException) as e:
        predict(model, Mock())
        assert e.status_code == 500
