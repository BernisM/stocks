"""
Système de scoring 0-100 :
  - Tendance     : 25 pts
  - Momentum     : 25 pts
  - Volume / OBV : 20 pts
  - Ichimoku     : 15 pts
  - Boost ML     : -15 à +15 pts
"""
from __future__ import annotations


def compute_score(ind: dict, ml_prob: float | None = None) -> tuple[int, int, int, str]:
    """
    Retourne (score_base, ml_boost, score_final, ranking).
    `ind` = dict issu de get_last_row().
    `ml_prob` = probabilité entre 0 et 1 (None si modèle absent).
    """
    score = 0

    # ── 1. Tendance (25 pts) ──────────────────────────────────────────────────
    sma50  = ind.get("SMA50",  0) or 0
    sma200 = ind.get("SMA200", 0) or 0
    ema50  = ind.get("EMA50",  0) or 0
    close  = ind.get("Close",  0) or 0

    if sma50 > sma200:          score += 10  # golden cross
    if close > ema50:           score += 8   # prix au-dessus EMA50
    if close > sma200:          score += 7   # prix au-dessus SMA200

    # ── 2. Momentum (25 pts) ─────────────────────────────────────────────────
    rsi       = ind.get("RSI",       50) or 50
    macd_hist = ind.get("MACD_hist", 0)  or 0

    if 50 <= rsi <= 70:   score += 12   # zone haussière idéale
    elif rsi < 30:        score += 15   # survente → rebond possible
    elif 30 <= rsi < 50:  score += 5    # neutre/bas
    elif rsi > 70:        score -= 5    # surachat → risque

    if macd_hist > 0:     score += 13   # histogramme positif

    # ── 3. Volume & OBV (20 pts) ─────────────────────────────────────────────
    vol_ratio = ind.get("Vol_ratio", 1) or 1
    obv_slope = ind.get("OBV_slope", 0) or 0

    if vol_ratio >= 1.5:  score += 10
    elif vol_ratio >= 1:  score += 5

    if obv_slope > 0:     score += 10

    # ── 4. Ichimoku (15 pts) ─────────────────────────────────────────────────
    tenkan     = ind.get("Tenkan",    0) or 0
    kijun      = ind.get("Kijun",     0) or 0
    above_cloud = ind.get("Above_cloud", False)

    if tenkan > kijun:   score += 8
    if above_cloud:      score += 7

    score_base = max(0, min(85, score))

    # ── 5. Boost ML (-15 à +15 pts) ──────────────────────────────────────────
    if ml_prob is None:
        ml_boost = 0
    elif ml_prob >= 0.70:  ml_boost = 15
    elif ml_prob >= 0.60:  ml_boost = 8
    elif ml_prob >= 0.50:  ml_boost = 3
    elif ml_prob >= 0.40:  ml_boost = 0
    else:                  ml_boost = -15

    score_final = max(0, min(100, score_base + ml_boost))

    # ── Ranking ───────────────────────────────────────────────────────────────
    if score_final >= 75:   ranking = "Strong Buy"
    elif score_final >= 55: ranking = "Buy"
    elif score_final >= 35: ranking = "Neutral"
    else:                   ranking = "Avoid"

    return score_base, ml_boost, score_final, ranking


RANKING_EMOJI = {
    "Strong Buy": "🔥",
    "Buy":        "🟢",
    "Neutral":    "⚪",
    "Avoid":      "🔴",
}
