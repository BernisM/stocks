"""
Modèle ML : RandomForestClassifier
Label : hausse > 2 % dans les 10 prochains jours de bourse
Features : 17 indicateurs techniques + régimes de marché normalisés
Ré-entraîné chaque matin après sync.
"""
from __future__ import annotations
import gc
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import ML_MODEL_PATH, ML_SCALER_PATH, XGB_MODEL_PATH

logger = logging.getLogger(__name__)

FEATURES = [
    "RSI", "MACD_hist", "ATR_pct", "BB_pct",
    "Vol_ratio", "OBV_slope",
    "EMA50_cross",        # binaire : Close > EMA50
    "Golden_cross_bool",  # binaire : SMA50 > SMA200
    "Ichimoku_bull",      # binaire : Tenkan > Kijun
    "Price_vs_SMA200",    # (Close/SMA200 - 1) × 100
    # Régimes de marché
    "ADX",                # force de la tendance (14)
    "SMA200_slope",       # pente SMA200 sur 10 jours (%)
    "ATR_pct_rank",       # rang percentile ATR sur 50 jours
    "BB_zscore",          # distance à la BB_middle en écarts-types
    "regime_trend",       # binaire : ADX > 25
    "regime_bull",        # binaire : SMA200_slope > 0
    "regime_vol_high",    # binaire : ATR_pct_rank > 70
]

_model:     RandomForestClassifier | None = None
_scaler:    StandardScaler | None = None
_xgb_model: XGBClassifier | None = None


def _load() -> bool:
    global _model, _scaler
    if not (os.path.exists(ML_MODEL_PATH) and os.path.exists(ML_SCALER_PATH)):
        return False
    try:
        model  = joblib.load(ML_MODEL_PATH)
        scaler = joblib.load(ML_SCALER_PATH)
        # Vérifie que le modèle a été entraîné avec le bon nombre de features
        n = getattr(scaler, "n_features_in_", None)
        if n is not None and n != len(FEATURES):
            logger.warning(
                f"Modèle obsolète ignoré : scaler attend {n} features, "
                f"code attend {len(FEATURES)}. Relancez /admin/train-ml."
            )
            return False
        _model, _scaler = model, scaler
        return True
    except Exception as e:
        logger.warning(f"Échec chargement modèle ML : {e}")
        return False


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["RSI"]               = df.get("RSI", pd.Series(dtype=float))
    feat["MACD_hist"]         = df.get("MACD_hist", pd.Series(dtype=float))
    feat["ATR_pct"]           = df.get("ATR_pct", pd.Series(dtype=float))
    feat["BB_pct"]            = df.get("BB_pct", pd.Series(dtype=float))
    feat["Vol_ratio"]         = df.get("Vol_ratio", pd.Series(dtype=float))
    feat["OBV_slope"]         = df.get("OBV_slope", pd.Series(dtype=float))
    feat["EMA50_cross"]       = (df["Close"] > df.get("EMA50",  df["Close"])).astype(int)
    feat["Golden_cross_bool"] = (df.get("SMA50", df["Close"]) > df.get("SMA200", df["Close"])).astype(int)
    feat["Ichimoku_bull"]     = (df.get("Tenkan", df["Close"]) > df.get("Kijun", df["Close"])).astype(int)
    sma200 = df.get("SMA200", df["Close"]).replace(0, np.nan)
    feat["Price_vs_SMA200"]   = (df["Close"] / sma200 - 1) * 100
    # Régimes
    feat["ADX"]               = df.get("ADX", pd.Series(dtype=float))
    feat["SMA200_slope"]      = df.get("SMA200_slope", pd.Series(dtype=float))
    feat["ATR_pct_rank"]      = df.get("ATR_pct_rank", pd.Series(dtype=float))
    feat["BB_zscore"]         = df.get("BB_zscore", pd.Series(dtype=float))
    feat["regime_trend"]      = df.get("regime_trend", pd.Series(dtype=float))
    feat["regime_bull"]       = df.get("regime_bull", pd.Series(dtype=float))
    feat["regime_vol_high"]   = df.get("regime_vol_high", pd.Series(dtype=float))
    return feat


def train(dfs: list[pd.DataFrame]) -> dict:
    """
    Entraîne le modèle sur une liste de DataFrames (un par action).
    Retourne les métriques.
    """
    global _model, _scaler
    all_X, all_y = [], []

    for df in dfs:
        if len(df) < 60:
            continue
        feat = _build_features(df)
        label = (df["Close"].shift(-10) / df["Close"] - 1 > 0.02).astype(int)

        data = feat.join(label.rename("label"))
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(data) < 30:
            continue

        all_X.append(data[FEATURES])
        all_y.append(data["label"])

    if not all_X:
        logger.warning("Not enough data to train ML model")
        return {}

    X = pd.concat(all_X).values
    y = pd.concat(all_y).values
    del all_X, all_y
    gc.collect()

    n_samples = len(X)
    # Découpage chrono : 80% train, 20% test
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    del X, y
    gc.collect()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── RandomForest ─────────────────────────────────────────────────────────
    # n_jobs=1 : pas de fork multiprocessing → pas de duplication mémoire (Render 512 MB)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=20,
        n_jobs=1, random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # ── XGBoost (tree_method=hist → mémoire réduite) ─────────────────────────
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=1, tree_method="hist",
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_pred  = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    del X_train, X_test, y_train
    gc.collect()

    metrics = {
        "accuracy":     round(accuracy_score(y_test, rf_pred)  * 100, 1),
        "auc":          round(roc_auc_score(y_test, rf_proba)  * 100, 1),
        "xgb_accuracy": round(accuracy_score(y_test, xgb_pred) * 100, 1),
        "xgb_auc":      round(roc_auc_score(y_test, xgb_proba) * 100, 1),
        "n_samples": n_samples,
        "n_train":   split,
    }
    logger.info(
        f"RF  — accuracy {metrics['accuracy']}% | AUC {metrics['auc']}%  |  "
        f"XGB — accuracy {metrics['xgb_accuracy']}% | AUC {metrics['xgb_auc']}%"
    )

    os.makedirs(os.path.dirname(ML_MODEL_PATH), exist_ok=True)
    joblib.dump(rf,     ML_MODEL_PATH)
    joblib.dump(scaler, ML_SCALER_PATH)
    joblib.dump(xgb,    XGB_MODEL_PATH)

    import json
    feat_imp = dict(zip(FEATURES, [round(v, 4) for v in rf.feature_importances_]))
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    with open(ML_MODEL_PATH.replace(".pkl", "_feature_importance.json"), "w") as f:
        json.dump(feat_imp_sorted, f, indent=2)

    global _model, _scaler, _xgb_model
    _model, _scaler, _xgb_model = rf, scaler, xgb

    return metrics


def predict(df: pd.DataFrame) -> float | None:
    """Retourne la probabilité d'achat (0-1) pour la dernière ligne du DataFrame."""
    global _model, _scaler
    if _model is None:
        if not _load():
            return None

    feat = _build_features(df)
    row  = feat.iloc[[-1]].replace([np.inf, -np.inf], np.nan)
    if row.isnull().any(axis=1).iloc[0]:
        return None

    try:
        X    = _scaler.transform(row[FEATURES].values)
        prob = float(_model.predict_proba(X)[0][1])
        return prob
    except Exception:
        _model = None
        _scaler = None
        return None


def predict_xgb(df: pd.DataFrame) -> float | None:
    """Probabilité XGBoost pour la dernière ligne (comparaison, non utilisé dans le score)."""
    global _xgb_model, _scaler
    if _xgb_model is None:
        if not os.path.exists(XGB_MODEL_PATH):
            return None
        try:
            _xgb_model = joblib.load(XGB_MODEL_PATH)
        except Exception:
            return None
    if _scaler is None:
        if not _load():
            return None

    feat = _build_features(df)
    row  = feat.iloc[[-1]].replace([np.inf, -np.inf], np.nan)
    if row.isnull().any(axis=1).iloc[0]:
        return None
    try:
        X    = _scaler.transform(row[FEATURES].values)
        return float(_xgb_model.predict_proba(X)[0][1])
    except Exception:
        return None


def load_metrics() -> dict:
    """Retourne les métriques sauvegardées si disponibles."""
    metrics_path = ML_MODEL_PATH.replace(".pkl", "_metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    import json
    with open(metrics_path) as f:
        return json.load(f)


def save_metrics(metrics: dict) -> None:
    import json
    path = ML_MODEL_PATH.replace(".pkl", "_metrics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f)


def load_feature_importance() -> dict:
    """Retourne l'importance des features si disponible (dict trié)."""
    import json
    path = ML_MODEL_PATH.replace(".pkl", "_feature_importance.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
