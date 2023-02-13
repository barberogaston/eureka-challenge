from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi import HTTPException

from housing_inference.api.parse import parse_params


def test_parse(features_dtypes):
    request = Mock()
    request.query_params = {"bar": 1, "baz": 2.3}

    result = parse_params(request, features_dtypes)

    assert isinstance(result, pd.DataFrame)
    assert sorted(result.columns) == sorted(["bar", "baz"])
    assert result.iloc[0].to_dict() == {"bar": 1, "baz": 2.3}


def test_parse_with_missing_features(features_dtypes):
    request = Mock()
    request.query_params = {"bar": 1}

    with pytest.raises(HTTPException) as e:
        parse_params(request, features_dtypes)
        assert e.status_code == 400
        assert e.detail == "Missing the following features: baz"
