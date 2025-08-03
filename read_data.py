import numpy as np

import pandas as pd
from glob import glob
import os
# ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
import xgboost as xgb

def load_field_data(data_dir='data_example/field_data', gases=None):
    if gases is None:
        gases = ['NO2', 'CO', 'O3', 'SO2', 'H2S']
    files = glob(os.path.join(data_dir, '*.csv'))
    if not files:
        print(f"[load_field_data] Нет файлов в {data_dir}")
        return pd.DataFrame()
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)
    
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    for gas in gases:
        signal_col = f"{gas}op1"
        temp_col = f"{gas}t"
        if signal_col in df:
            df[f'{signal_col}_grad'] = df[signal_col].diff()
        if temp_col in df:
            df[f'{temp_col}_grad'] = df[temp_col].diff()
    for hum_col in ['MH', 'g1_mh', 'g2_mh', 'humidity', 'RH', 'rh_thc', 'righth', 'lefth']:
        if hum_col in df:
            df[f'{hum_col}_grad'] = df[hum_col].diff()
    return df


def load_lab_data(data_dir='data_example/lab_data', gases=None):
    if gases is None:
        gases = ['NO2', 'CO', 'O3', 'SO2', 'H2S']

    files = glob(os.path.join(data_dir, '*_stat.csv'))
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('date')

    # Добавляем градиенты по каждому газу
    for gas in gases:
        signal_col = f"{gas}op1"
        temp_col = f"{gas}t"
        if signal_col in df:
            df[f'{signal_col}_grad'] = df[signal_col].diff()
        if temp_col in df:
            df[f'{temp_col}_grad'] = df[temp_col].diff()

    df['MH_grad'] = df['MH'].diff() if 'MH' in df else np.nan
    return df

def compute_baseline_mask(
    df,
    gas,
    signal_col=None,
    baseline_eps=0.01,
    cross_gases=None,     
    use_iqr=True,
    use_isolation=False,
    contamination=None,
    verbose=False
):
    """
    Формируем маску baseline: сначала просто по концентрации, потом фильтруем выбросы по IQR и/или IsolationForest.
    """
    from sklearn.ensemble import IsolationForest

    if signal_col is None:
        signal_col = f"{gas}op1"
    temp_col = f"{gas}t"

    valid_mask = df[signal_col].notna()
    bl_mask = (df[gas].fillna(0) < baseline_eps) & valid_mask

    if cross_gases:
        for cg in cross_gases:
            if cg in df.columns:
                bl_mask = bl_mask & (df[cg].fillna(0).abs() < baseline_eps)

    #  Фильтруем baseline по IQR 
    if use_iqr:
        q1 = df.loc[bl_mask, signal_col].quantile(0.25)
        q3 = df.loc[bl_mask, signal_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = df[signal_col].between(lower, upper)
        bl_mask = bl_mask & iqr_mask

    #    Isolation Forest 
    if use_isolation:
        features = [signal_col]
        if temp_col in df: features.append(temp_col)
        if 'MH' in df: features.append('MH')
        if f"{temp_col}_grad" in df: features.append(f"{temp_col}_grad")
        if 'MH_grad' in df: features.append('MH_grad')
        iso_df = df.loc[bl_mask, features].dropna()
        if not iso_df.empty:
            # contamination  by  IQR 
            if contamination is None and use_iqr:
                extreme = (df.loc[bl_mask, signal_col] < lower) | (df.loc[bl_mask, signal_col] > upper)
                contamination_est = min(max(extreme.sum() / len(extreme), 0.001), 0.2)
            else:
                contamination_est = contamination if contamination is not None else 0.02
            if verbose:
                print(f"{gas}: estimated contamination = {contamination_est:.4f}")
            iso = IsolationForest(contamination=contamination_est, random_state=42)
            outliers = iso.fit_predict(iso_df)
            bad_idx = iso_df.index[outliers == -1]
            bl_mask.loc[bad_idx] = False

    if verbose:
        print(f"{gas}: {bl_mask.sum()} baseline (после фильтрации)")

    return bl_mask

def add_bl_stat_masks(df, gases=None, std_thr=0.1, window=5, baseline_eps=0.001, use_isolation=True, contamination=None, verbose=True, cross_gases_map=None):
    if gases is None:
        gases = ['NO2', 'CO', 'O3', 'SO2', 'H2S']

    for gas in gases:
        signal_col = f"{gas}op1"
        cross_gases = cross_gases_map.get(gas, []) if cross_gases_map else []

        if gas in df.columns and signal_col in df.columns:
            df[f"{signal_col}_bl"] = compute_baseline_mask(
                df, gas,
                baseline_eps=baseline_eps,
                cross_gases=cross_gases,  
                use_iqr=True,
                use_isolation=use_isolation,
                contamination=contamination,
                verbose=verbose
            )           

            rolling_std = df[signal_col].rolling(window, center=True).std()
            df[f"{signal_col}_stat"] = (rolling_std < std_thr).fillna(False) & (~df[f"{signal_col}_bl"])

            if verbose:               
                n_total = len(df)
                n_bl = df[f'{signal_col}_bl'].sum()
                n_stat = df[f'{signal_col}_stat'].sum()
                print(f"{gas}: baseline = {n_bl}, stat = {n_stat}, всего = {n_total}")            
        else:
            df[f"{signal_col}_bl"] = False
            df[f"{signal_col}_stat"] = False
            if verbose:
                print(f"{gas}: пропущены (нет данных)")

    return df

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
