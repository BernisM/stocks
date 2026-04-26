from __future__ import annotations
import logging
import time

import yfinance as yf
from sqlalchemy.orm import Session

from .models import AnalysisResult, Stock

log = logging.getLogger(__name__)


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _score_pe(pe) -> int:
    """P/E forward ou trailing. yfinance: valeur numérique (ex: 22.5). Max 30 pts."""
    if pe is None or pe <= 0:
        return 12  # neutre — beaucoup de growth stocks ont PE négatif
    if pe < 10:  return 30
    if pe < 15:  return 25
    if pe < 25:  return 20
    if pe < 35:  return 13
    return 5


def _score_pb(pb) -> int:
    """P/B ratio. Max 20 pts."""
    if pb is None or pb <= 0:
        return 8
    if pb < 1:  return 20
    if pb < 2:  return 17
    if pb < 4:  return 12
    if pb < 8:  return 6
    return 3


def _score_roe(roe) -> int:
    """ROE (yfinance: fraction, ex: 0.18 = 18%). Max 25 pts."""
    if roe is None:
        return 10
    pct = roe * 100
    if pct > 25:  return 25
    if pct > 15:  return 20
    if pct > 8:   return 14
    if pct > 0:   return 7
    return 0


def _score_debt(de) -> int:
    """Debt/Equity. yfinance retourne en % (ex: 150 = 1.5×). Max 15 pts."""
    if de is None:
        return 7
    if de < 30:   return 15
    if de < 80:   return 12
    if de < 150:  return 9
    if de < 300:  return 4
    return 0


def _score_growth(growth) -> int:
    """Revenue growth (yfinance: fraction, ex: 0.12 = 12%). Max 10 pts."""
    if growth is None:
        return 4
    pct = growth * 100
    if pct > 20:  return 10
    if pct > 10:  return 8
    if pct > 3:   return 5
    if pct > 0:   return 3
    return 0


# ── Calcul du score ───────────────────────────────────────────────────────────

def compute_fundamental_score(info: dict) -> tuple[int, dict]:
    """
    Retourne (score 0-100, métriques brutes).
    Score = PE(30) + PB(20) + ROE(25) + D/E(15) + Growth(10)
    """
    pe     = info.get("forwardPE") or info.get("trailingPE")
    pb     = info.get("priceToBook")
    roe    = info.get("returnOnEquity")
    de     = info.get("debtToEquity")
    growth = info.get("revenueGrowth")

    score = (
        _score_pe(pe)
        + _score_pb(pb)
        + _score_roe(roe)
        + _score_debt(de)
        + _score_growth(growth)
    )  # max = 100

    metrics = {
        "pe":     round(pe, 1)           if pe     else None,
        "pb":     round(pb, 2)           if pb     else None,
        "roe":    round(roe * 100, 1)    if roe    else None,
        "de":     round(de, 1)           if de     else None,
        "growth": round(growth * 100, 1) if growth else None,
    }
    return score, metrics


# ── Mise à jour en base ───────────────────────────────────────────────────────

def update_fundamentals(db: Session) -> None:
    """
    Fetch les fondamentaux via yfinance pour tous les stocks
    et met à jour la dernière AnalysisResult de chaque stock.
    Durée estimée : ~667 × 0.5s ≈ 6 min.
    """
    stocks = db.query(Stock).all()
    log.info(f"[fundamentals] Fetch pour {len(stocks)} stocks…")
    updated = errors = 0

    for stock in stocks:
        if stock.market in ("COMMODITIES", "CRYPTO"):
            continue  # pas de fondamentaux pour les futures et cryptos
        try:
            # Retry une fois si rate limited
            try:
                info = yf.Ticker(stock.ticker).info
            except Exception as e:
                if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                    time.sleep(60)
                    info = yf.Ticker(stock.ticker).info
                else:
                    raise
            score, metrics = compute_fundamental_score(info)

            name = info.get("longName") or info.get("shortName")
            if name and not stock.name:
                stock.name = name

            sector = info.get("sector")
            if sector:
                stock.sector = sector

            # Métriques avancées
            peg      = info.get("pegRatio") or info.get("trailingPegRatio")
            ev_ebitda_raw = info.get("enterpriseToEbitda")
            ev_raw   = info.get("enterpriseValue")
            ebit_raw = info.get("ebit")
            fcf_raw  = info.get("freeCashflow")

            ev_ebit_val = None
            if ev_raw and ebit_raw and ebit_raw > 0:
                ev_ebit_val = round(ev_raw / ebit_raw, 1)

            result = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.stock_id == stock.id)
                .order_by(AnalysisResult.date.desc())
                .first()
            )
            if result:
                result.fundamental_score = score
                result.pe_ratio          = metrics["pe"]
                result.pb_ratio          = metrics["pb"]
                result.roe               = metrics["roe"]
                result.debt_equity       = metrics["de"]
                result.rev_growth        = metrics["growth"]
                result.peg_ratio         = round(peg, 2)        if peg       else None
                result.ev_ebit           = ev_ebit_val
                result.ev_ebitda         = round(ev_ebitda_raw, 1) if ev_ebitda_raw else None
                result.fcf               = fcf_raw  # valeur brute en devise de cotation
                # Composite : 65 % technique + 35 % fondamental
                tech = result.score_final or 0
                result.score_composite   = round(tech * 0.65 + score * 0.35)
                db.commit()
                updated += 1

            time.sleep(1.0)  # politesse envers l'API Yahoo (évite le rate limiting)

        except Exception as e:
            db.rollback()
            errors += 1
            log.warning(f"[{stock.ticker}] fundamentals error: {e}")

    log.info(f"[fundamentals] Terminé — {updated} mis à jour, {errors} erreurs")
