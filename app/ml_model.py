"""
Ensemble RF+XGB+LGB — 4 modèles séparés par groupe de marché.
  EUROPE (CAC40 + SBF120) / US (SP500) / CRYPTO / COMMO (COMMODITIES)
Label : hausse > 1×ATR_pct dans 10 jours (dynamique par actif, clip 0.5%–8%)
Features : 22 techniques + 5 fondamentaux (EUROPE + US uniquement)
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
from lightgbm import LGBMClassifier

logger  = logging.getLogger(__name__)
ML_DIR  = "./ml_models"

# ── Groupes de marchés ────────────────────────────────────────────────────────

GROUPS: dict[str, set[str]] = {
    "EUROPE": {"CAC40", "SBF120", "EURONEXT_GROWTH"},
    "US":     {"SP500", "NASDAQ"},
    "CRYPTO": {"CRYPTO"},
    "COMMO":  {"COMMODITIES"},
}

def get_group(market: str) -> str:
    for grp, markets in GROUPS.items():
        if market in markets:
            return grp
    return "US"

# ── Features ──────────────────────────────────────────────────────────────────

FEATURES_TECH: list[str] = [
    "RSI", "MACD_hist", "ATR_pct", "BB_pct",
    "Vol_ratio", "OBV_slope",
    "EMA50_cross", "Golden_cross_bool", "Ichimoku_bull", "Price_vs_SMA200",
    "ADX", "SMA200_slope", "ATR_pct_rank", "BB_zscore",
    "regime_trend", "regime_bull", "regime_vol_high",
    "Return_1d", "Return_5d", "Price_vs_High",
    "RSI_slope", "MACD_accel",
]

FEATURES_FUND: list[str] = [
    "fund_pe", "fund_roe", "fund_de", "fund_growth", "fund_score",
]

FEATURES_BY_GROUP: dict[str, list[str]] = {
    "EUROPE": FEATURES_TECH + FEATURES_FUND,
    "US":     FEATURES_TECH + FEATURES_FUND,
    "CRYPTO": FEATURES_TECH,
    "COMMO":  FEATURES_TECH,
}

# Alias gardé pour compatibilité (scheduler _keep)
FEATURES = FEATURES_TECH

# ── État global (chargé à la demande par groupe) ──────────────────────────────

_state: dict[str, dict] = {}   # group → {rf, xgb, lgb, scaler}


def _paths(group: str) -> dict[str, str]:
    g = group.lower()
    return {
        "rf":     f"{ML_DIR}/rf_{g}.pkl",
        "xgb":    f"{ML_DIR}/xgb_{g}.pkl",
        "lgb":    f"{ML_DIR}/lgb_{g}.pkl",
        "scaler": f"{ML_DIR}/scaler_{g}.pkl",
    }


def _load_group(group: str) -> bool:
    p = _paths(group)
    if not (os.path.exists(p["rf"]) and os.path.exists(p["scaler"])):
        return False
    try:
        scaler     = joblib.load(p["scaler"])
        n_expected = len(FEATURES_BY_GROUP[group])
        n_actual   = getattr(scaler, "n_features_in_", None)
        if n_actual is not None and n_actual != n_expected:
            logger.warning(
                f"[{group}] Scaler obsolète : attend {n_actual} features, "
                f"code attend {n_expected}. Relancez /admin/train-ml."
            )
            return False
        models: dict = {
            "rf":     joblib.load(p["rf"]),
            "scaler": scaler,
            "xgb":    None,
            "lgb":    None,
        }
        for key in ("xgb", "lgb"):
            if os.path.exists(p[key]):
                try:
                    models[key] = joblib.load(p[key])
                except Exception:
                    pass
        _state[group] = models
        logger.info(f"[{group}] Modèles chargés ({n_expected} features)")
        return True
    except Exception as e:
        logger.warning(f"[{group}] Chargement échoué : {e}")
        return False


# ── Construction des features ─────────────────────────────────────────────────

def _build_features(df: pd.DataFrame, group: str = "US") -> pd.DataFrame:
    feat  = pd.DataFrame(index=df.index)
    close = df["Close"]

    feat["RSI"]               = df.get("RSI",      pd.Series(dtype=float))
    feat["MACD_hist"]         = df.get("MACD_hist", pd.Series(dtype=float))
    feat["ATR_pct"]           = df.get("ATR_pct",   pd.Series(dtype=float))
    feat["BB_pct"]            = df.get("BB_pct",    pd.Series(dtype=float))
    feat["Vol_ratio"]         = df.get("Vol_ratio", pd.Series(dtype=float))
    feat["OBV_slope"]         = df.get("OBV_slope", pd.Series(dtype=float))

    feat["EMA50_cross"]       = (close > df.get("EMA50",  close)).astype(int)
    feat["Golden_cross_bool"] = (df.get("SMA50",  close) > df.get("SMA200", close)).astype(int)
    feat["Ichimoku_bull"]     = (df.get("Tenkan", close) > df.get("Kijun",  close)).astype(int)
    sma200 = df.get("SMA200", close).replace(0, np.nan)
    feat["Price_vs_SMA200"]   = (close / sma200 - 1) * 100

    feat["ADX"]               = df.get("ADX",          pd.Series(dtype=float))
    feat["SMA200_slope"]      = df.get("SMA200_slope",  pd.Series(dtype=float))
    feat["ATR_pct_rank"]      = df.get("ATR_pct_rank",  pd.Series(dtype=float))
    feat["BB_zscore"]         = df.get("BB_zscore",     pd.Series(dtype=float))
    feat["regime_trend"]      = df.get("regime_trend",  pd.Series(dtype=float))
    feat["regime_bull"]       = df.get("regime_bull",   pd.Series(dtype=float))
    feat["regime_vol_high"]   = df.get("regime_vol_high", pd.Series(dtype=float))

    feat["Return_1d"]         = close.pct_change(1) * 100
    feat["Return_5d"]         = close.pct_change(5) * 100
    high_200                  = close.rolling(200, min_periods=20).max().replace(0, np.nan)
    feat["Price_vs_High"]     = (close / high_200 - 1) * 100

    rsi                       = df.get("RSI", pd.Series(dtype=float, index=df.index))
    feat["RSI_slope"]         = rsi.diff(5)
    macd_h                    = df.get("MACD_hist", pd.Series(dtype=float, index=df.index))
    feat["MACD_accel"]        = macd_h.diff(1).apply(
        lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
    )

    # Fondamentaux — EUROPE + US uniquement
    if group in ("EUROPE", "US"):
        for col in FEATURES_FUND:
            feat[col] = df.get(col, pd.Series(0.0, index=df.index))

    return feat


# ── Label dynamique ───────────────────────────────────────────────────────────

def _make_label(df: pd.DataFrame) -> pd.Series:
    """Hausse > 1×ATR_pct dans 10 jours. Seuil clippé entre 0.5% et 8%."""
    close     = df["Close"]
    atr_pct   = df.get("ATR_pct", pd.Series(2.0, index=df.index))
    threshold = atr_pct.clip(0.5, 8.0) / 100
    return (close.shift(-10) / close - 1 > threshold).astype(int)


# ── Entraînement d'un groupe ──────────────────────────────────────────────────

def _train_group(group: str, dfs: list[pd.DataFrame]) -> dict:
    features = FEATURES_BY_GROUP[group]
    all_X, all_y = [], []

    # Cap par stock : max 80 lignes pour borner la RAM avant concat
    # (606 stocks × 80 rows × 27 features ≈ 40 MB max en float32)
    MAX_ROWS_PER_STOCK = 80

    for df in dfs:
        if len(df) < 60:
            continue
        feat  = _build_features(df, group)
        label = _make_label(df)

        data = feat[features].join(label.rename("label"))
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 30:
            continue

        # Garde uniquement les N dernières lignes pour limiter la RAM
        data = data.iloc[-MAX_ROWS_PER_STOCK:]

        all_X.append(data[features])
        all_y.append(data["label"])

    if not all_X:
        logger.warning(f"[{group}] Pas assez de données pour entraîner.")
        return {}

    X = pd.concat(all_X).astype(np.float32).values
    y = pd.concat(all_y).values
    del all_X, all_y
    gc.collect()

    n_samples = len(X)

    # Cap mémoire strict — Render free tier = 512 MB
    large     = n_samples > 10_000
    MAX_SAMP  = 10_000 if large else n_samples
    if n_samples > MAX_SAMP:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, MAX_SAMP, replace=False)
        idx.sort()
        X, y = X[idx], y[idx]
        n_samples = MAX_SAMP
        logger.info(f"[{group}] Sous-échantillonnage → {MAX_SAMP} obs")

    split      = int(n_samples * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    del X, y
    gc.collect()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    small    = n_samples < 3_000
    min_leaf = 5 if small else 20

    import json
    os.makedirs(ML_DIR, exist_ok=True)
    p = _paths(group)
    joblib.dump(scaler, p["scaler"])

    # ── RandomForest — sauvegarde puis libération RAM ─────────────────────────
    n_trees = 50 if large else 100
    rf = RandomForestClassifier(
        n_estimators=n_trees, max_depth=8, min_samples_leaf=min_leaf,
        class_weight="balanced", n_jobs=1, random_state=42,
    )
    rf.fit(X_tr, y_tr)
    rf_proba = rf.predict_proba(X_te)[:, 1]
    rf_pred  = (rf_proba >= 0.5).astype(int)
    imp = dict(sorted(
        zip(features, [round(v, 4) for v in rf.feature_importances_]),
        key=lambda x: x[1], reverse=True,
    ))
    joblib.dump(rf, p["rf"])
    del rf; gc.collect()

    # ── LightGBM — sauvegarde puis libération RAM ─────────────────────────────
    n_lgb   = 100 if large else (200 if small else 300)
    lgb_clf = LGBMClassifier(
        n_estimators=n_lgb, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=min_leaf, n_jobs=1, random_state=42, verbose=-1,
    )
    lgb_clf.fit(X_tr, y_tr)
    lgb_proba = lgb_clf.predict_proba(X_te)[:, 1]
    lgb_pred  = (lgb_proba >= 0.5).astype(int)
    joblib.dump(lgb_clf, p["lgb"])
    del lgb_clf; gc.collect()

    # ── XGBoost — uniquement pour petits groupes (RAM) ────────────────────────
    if not large:
        n_xgb = 150 if small else 200
        xgb = XGBClassifier(
            n_estimators=n_xgb, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=1, tree_method="hist", eval_metric="logloss",
            verbosity=0, random_state=42,
        )
        xgb.fit(X_tr, y_tr)
        xgb_proba = xgb.predict_proba(X_te)[:, 1]
        xgb_pred  = (xgb_proba >= 0.5).astype(int)
        joblib.dump(xgb, p["xgb"])
        del xgb; gc.collect()
        ens_proba = (rf_proba + xgb_proba + lgb_proba) / 3
    else:
        # XGB désactivé pour économiser la RAM — ensemble = moyenne RF+LGB
        xgb_proba = rf_proba.copy()   # copie pour affichage métriques seulement
        xgb_pred  = rf_pred.copy()
        ens_proba = (rf_proba + lgb_proba) / 2

    del X_tr, y_tr
    gc.collect()

    ens_pred  = (ens_proba >= 0.5).astype(int)

    metrics = {
        "accuracy":          round(accuracy_score(y_te, rf_pred)   * 100, 1),
        "auc":               round(roc_auc_score(y_te,  rf_proba)  * 100, 1),
        "xgb_accuracy":      round(accuracy_score(y_te, xgb_pred)  * 100, 1),
        "xgb_auc":           round(roc_auc_score(y_te,  xgb_proba) * 100, 1),
        "lgb_accuracy":      round(accuracy_score(y_te, lgb_pred)  * 100, 1),
        "lgb_auc":           round(roc_auc_score(y_te,  lgb_proba) * 100, 1),
        "ensemble_accuracy": round(accuracy_score(y_te, ens_pred)  * 100, 1),
        "ensemble_auc":      round(roc_auc_score(y_te,  ens_proba) * 100, 1),
        "n_samples":         n_samples,
        "n_features":        len(features),
    }
    logger.info(
        f"[{group}] RF {metrics['accuracy']}%/{metrics['auc']}%  "
        f"XGB {metrics['xgb_accuracy']}%/{metrics['xgb_auc']}%  "
        f"LGB {metrics['lgb_accuracy']}%/{metrics['lgb_auc']}%  "
        f"Ensemble {metrics['ensemble_accuracy']}%/{metrics['ensemble_auc']}%  "
        f"({n_samples} obs, {len(features)} features)"
    )

    with open(f"{ML_DIR}/feature_importance_{group.lower()}.json", "w") as f:
        json.dump(imp, f, indent=2)

    # Les modèles ont déjà été supprimés après chaque sauvegarde disque
    _state.pop(group, None)
    del scaler
    gc.collect()

    return metrics


# ── Entraînement global ───────────────────────────────────────────────────────

def train(dfs_by_group: dict[str, list[pd.DataFrame]]) -> dict:
    """Entraîne un ensemble RF+XGB+LGB par groupe de marché."""
    all_metrics: dict[str, dict] = {}
    total_samples = 0

    for group, dfs in dfs_by_group.items():
        if not dfs:
            continue
        logger.info(f"=== Entraînement {group} ({len(dfs)} actifs) ===")
        m = _train_group(group, dfs)
        if m:
            all_metrics[group] = m
            total_samples += m.get("n_samples", 0)

    if not all_metrics:
        return {}

    # Métriques agrégées (moyennes pondérées par n_samples) pour le dashboard
    def _wavg(key: str) -> float:
        total = sum(all_metrics[g]["n_samples"] for g in all_metrics)
        if not total:
            return 0.0
        return round(sum(
            all_metrics[g][key] * all_metrics[g]["n_samples"]
            for g in all_metrics if key in all_metrics[g]
        ) / total, 1)

    summary = {
        "accuracy":          _wavg("accuracy"),
        "auc":               _wavg("auc"),
        "xgb_accuracy":      _wavg("xgb_accuracy"),
        "xgb_auc":           _wavg("xgb_auc"),
        "lgb_accuracy":      _wavg("lgb_accuracy"),
        "lgb_auc":           _wavg("lgb_auc"),
        "ensemble_accuracy": _wavg("ensemble_accuracy"),
        "ensemble_auc":      _wavg("ensemble_auc"),
        "n_samples":         total_samples,
        "groups":            all_metrics,
    }
    return summary


# ── Prédiction ────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame, market: str = "SP500") -> float | None:
    """Probabilité ensemble (RF+XGB+LGB) pour la dernière ligne du df."""
    group = get_group(market)

    if group not in _state:
        if not _load_group(group):
            return None

    st       = _state[group]
    features = FEATURES_BY_GROUP[group]
    feat     = _build_features(df, group)

    row = feat[features].iloc[[-1]].replace([np.inf, -np.inf], np.nan)
    if row.isnull().any(axis=1).iloc[0]:
        return None

    try:
        X     = st["scaler"].transform(row.values)
        probs = [float(st["rf"].predict_proba(X)[0][1])]
        for key in ("xgb", "lgb"):
            if st.get(key):
                try:
                    probs.append(float(st[key].predict_proba(X)[0][1]))
                except Exception:
                    pass
        return sum(probs) / len(probs)
    except Exception as e:
        logger.warning(f"predict() [{group}] : {e}")
        _state.pop(group, None)
        return None


# ── Métriques & feature importance ───────────────────────────────────────────

def load_metrics() -> dict:
    import json
    path = f"{ML_DIR}/metrics.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_metrics(metrics: dict) -> None:
    import json
    os.makedirs(ML_DIR, exist_ok=True)
    with open(f"{ML_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def load_feature_importance(group: str = "US") -> dict:
    import json
    path = f"{ML_DIR}/feature_importance_{group.lower()}.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
