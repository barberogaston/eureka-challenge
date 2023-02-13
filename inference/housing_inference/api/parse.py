import pandas as pd
from fastapi import Request, HTTPException


def parse_params(request: Request, features_dtypes: dict) -> pd.DataFrame:
    """
    Parse the query params and convert them into a pandas DataFrame,
    ready to be fed into the model.
    """
    params = {
        k: v for k, v in request.query_params.items() if k in features_dtypes
    }

    if missing := set(features_dtypes) - set(params):
        raise HTTPException(
            status_code=400,
            detail=f"Missing the following features: {', '.join(missing)}"
        )

    return pd.DataFrame.from_records([params]).astype(
        features_dtypes, errors="ignore", copy=False
    )
