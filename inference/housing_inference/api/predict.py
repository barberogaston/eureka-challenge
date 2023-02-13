import logging

import pandas as pd
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def predict(model, data: pd.DataFrame):
    try:
        return {"prediction": float(model.predict(data))}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) from e
