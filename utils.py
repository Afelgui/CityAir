import numpy as np

import pandas as pd
from glob import glob
import os
# ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
import xgboost as xgb



def get_model(name, **params):
    if name == 'linear': return LinearRegression(**params)
    if name == 'catboost': return CatBoostRegressor(verbose=0, random_seed=42, **params)
    if name == 'histgb': return HistGradientBoostingRegressor(random_state=42, **params)
    if name == 'xgb': return xgb.XGBRegressor(random_state=42, **params)
    raise ValueError(f"Неизвестная модель: {name}")

def get_importance(model, feat_names):
    if hasattr(model, "feature_importances_"):
        return dict(zip(feat_names, model.feature_importances_))
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim == 1:
            return dict(zip(feat_names, coefs))
        else:
            return dict(zip(feat_names, coefs.ravel()))
    else:
        return {}
