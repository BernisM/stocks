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
    elif rsi < 30:
        if macd_hist > 0: score += 8   # survente confirmée par rebond MACD
        # sinon 0 pts — couteau qui tombe
    elif 30 <= rsi < 50:  score += 5    # neutre/bas
    elif rsi > 70:        score -= 5    # surachat → risque

    if macd_hist > 0:     score += 13   # histogramme positif

    # ── 3. Volume & OBV (20 pts) ─────────────────────────────────────────────
    vol_ratio = ind.get("Vol_ratio", 1) or 1
    obv_slope = ind.get("OBV_slope", 0) or 0

    if vol_ratio >= 1.5:  score += 10
    elif vol_ratio >= 1.3: score += 5

    if obv_slope > 0:     score += 10

    # ── 4. Ichimoku (15 pts) ─────────────────────────────────────────────────
    tenkan     = ind.get("Tenkan",    0) or 0
    kijun      = ind.get("Kijun",     0) or 0
    above_cloud = ind.get("Above_cloud", False)

    if tenkan > kijun:   score += 8
    if above_cloud:      score += 7

    # ── 5. Régimes de marché (-10 à +5 pts) ──────────────────────────────────
    adx          = ind.get("ADX",          25) or 25
    regime_bull  = ind.get("regime_bull",   1)         # 1 = SMA200 haussière, 0 = baissière

    if adx < 20:
        score -= 5    # marché en range : signaux de tendance peu fiables
    elif adx > 30:
        score += 3    # forte tendance : signaux plus fiables

    if not regime_bull:
        score -= 5    # SMA200 décroissante → biais baissier global

    score_base = max(0, min(85, score))

    # ── 6. Boost ML (-15 à +15 pts) ──────────────────────────────────────────
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


# ── Hyper-Growth (détection licornes potentielles) ────────────────────────────

def compute_hyper_growth_score(
    ind: dict,
    rev_growth: float | None,
    debt_equity: float | None,
    fundamental_score: int | None,
    score_final: int,
    fcf: float | None,
    pb_ratio: float | None,
) -> int | None:
    """
    Score 0-100 — détecte les "futures licornes" (forte croissance + accumulation).
    Retourne None si non éligible.

    Eligibility (tous requis) :
      - rev_growth >= 15%
      - OBV slope > 0 (accumulation)
      - score_final >= 55 (pas de signal négatif)
      - fundamental_score disponible (exclut COMMO/CRYPTO)
      - debt_equity <= 300 (D/E <= 3×, exclut sur-endettés)

    Pondération :
      - 40 pts : croissance revenus
      - 25 pts : momentum technique (ADX + SMA200 slope)
      - 20 pts : accumulation volume (Vol_ratio + OBV slope)
      - 15 pts : valorisation / FCF
    """
    if rev_growth is None or rev_growth < 15:
        return None
    if fundamental_score is None:
        return None
    if score_final is None or score_final < 55:
        return None
    if debt_equity is not None and debt_equity > 300:
        return None
    obv_slope = ind.get("OBV_slope", 0) or 0
    if obv_slope <= 0:
        return None

    # 1. Croissance revenus (40 pts)
    if   rev_growth >= 50: growth_pts = 40
    elif rev_growth >= 30: growth_pts = 35
    elif rev_growth >= 25: growth_pts = 30
    elif rev_growth >= 20: growth_pts = 25
    else:                  growth_pts = 18   # 15–20%

    # 2. Momentum technique (25 pts)
    tech_pts     = 0
    adx          = ind.get("ADX",          0) or 0
    sma200_slope = ind.get("SMA200_slope", 0) or 0

    if   adx >= 30: tech_pts += 10
    elif adx >= 25: tech_pts += 7
    elif adx >= 20: tech_pts += 4

    if   sma200_slope > 0.5: tech_pts += 10
    elif sma200_slope > 0:   tech_pts += 6

    if obv_slope > 0:        tech_pts += 5
    tech_pts = min(25, tech_pts)

    # 3. Accumulation volume (20 pts)
    vol_pts   = 0
    vol_ratio = ind.get("Vol_ratio", 1) or 1
    if   vol_ratio >= 1.5: vol_pts += 12
    elif vol_ratio >= 1.3: vol_pts += 8
    elif vol_ratio >= 1.0: vol_pts += 4

    if   obv_slope > 0.5: vol_pts += 8
    elif obv_slope > 0:   vol_pts += 5
    vol_pts = min(20, vol_pts)

    # 4. Valorisation / croissance (15 pts)
    val_pts = 0
    if   fcf is not None and fcf > 0: val_pts += 8
    elif fcf is None:                 val_pts += 4
    # fcf < 0 → 0 pts

    if pb_ratio is not None:
        if   pb_ratio < 5 and rev_growth > 25: val_pts += 7
        elif pb_ratio < 10:                    val_pts += 4
    else:
        val_pts += 4
    val_pts = min(15, val_pts)

    return min(100, growth_pts + tech_pts + vol_pts + val_pts)
