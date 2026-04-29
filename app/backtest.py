from __future__ import annotations
"""
Backtesting de la stratégie de scoring sur données historiques.

Règles :
  - Entrée  : score >= 75 (Strong Buy) → achat au prix de clôture du jour suivant
  - Sortie  : stop-loss ATR franchi  OU  après MAX_HOLD jours
  - Position: une seule par action à la fois
  - Capital : 10 000 € fictifs par marché (pour normaliser)
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .config import ATR_STOP_MULTIPLIER
from .data_engine import get_dataframe
from .indicators import compute_indicators
from .models import AnalysisResult, Stock
from .scoring import compute_score

logger = logging.getLogger(__name__)

MAX_HOLD        = 20    # jours max avant vente forcée
SCORE_BUY       = 80    # seuil signal achat
MIN_FUNDAMENTAL = 40    # score fondamental minimum (None = pas encore calculé → autorisé)
TAKE_PROFIT     = 0.07  # +7 % → sortie gagnante anticipée


@dataclass
class Trade:
    ticker:      str
    market:      str
    entry_date:  datetime
    exit_date:   datetime
    entry_price: float
    exit_price:  float
    stop_loss:   float
    exit_reason: str   # "stop_loss" | "take_profit" | "timeout" | "end_of_data"
    regime:      str   = ""  # "trend_bull" | "trend_bear" | "range_bull" | "range_bear"
    sector:      str   = ""

    @property
    def pnl_pct(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def win(self) -> bool:
        return self.pnl_pct > 0


@dataclass
class MarketStats:
    market:        str
    n_trades:      int     = 0
    win_rate:      float   = 0.0
    avg_return:    float   = 0.0
    total_return:  float   = 0.0
    max_drawdown:  float   = 0.0
    sharpe:        float   = 0.0
    best_trade:    float   = 0.0
    worst_trade:   float   = 0.0
    avg_hold_days: float   = 0.0
    trades:        list    = field(default_factory=list)


def _run_stock_backtest(
    df: pd.DataFrame,
    ticker: str,
    market: str,
    fundamental_score: float | None = None,
    sector: str = "",
) -> list[Trade]:
    """Simule la stratégie sur un seul stock."""
    if len(df) < 60:
        return []
    # Filtre fondamental : exclure les actions avec un mauvais score (si disponible)
    if fundamental_score is not None and fundamental_score < MIN_FUNDAMENTAL:
        return []

    df = compute_indicators(df)
    trades: list[Trade] = []
    i = 0
    n = len(df)

    while i < n - 2:
        row    = df.iloc[i]
        ind    = row.to_dict()
        score_base, _, score_final, ranking = compute_score(ind)

        if score_final >= SCORE_BUY:
            # Entrée au prix de clôture du jour suivant
            entry_idx = i + 1
            if entry_idx >= n:
                break

            entry_row   = df.iloc[entry_idx]
            entry_price = entry_row["Close"]
            stop_loss   = entry_price - ATR_STOP_MULTIPLIER * (row.get("ATR") or entry_price * 0.05)
            take_profit = entry_price * (1 + TAKE_PROFIT)

            # Régime au moment de l'entrée
            _adx_raw   = row.get("ADX", 25)
            adx_val    = float(_adx_raw) if (_adx_raw is not None and _adx_raw == _adx_raw) else 25
            bull_val   = row.get("regime_bull", 1)
            regime     = ("trend" if adx_val > 25 else "range") + "_" + ("bull" if bull_val else "bear")

            exit_price  = entry_price
            exit_reason = "end_of_data"
            exit_idx    = min(entry_idx + MAX_HOLD, n - 1)

            for j in range(entry_idx + 1, exit_idx + 1):
                low_j   = df.iloc[j]["Low"]
                high_j  = df.iloc[j]["High"]
                close_j = df.iloc[j]["Close"]

                if low_j <= stop_loss:
                    exit_price  = stop_loss
                    exit_idx    = j
                    exit_reason = "stop_loss"
                    break

                if high_j >= take_profit:
                    exit_price  = take_profit
                    exit_idx    = j
                    exit_reason = "take_profit"
                    break

                if j == exit_idx:
                    exit_price  = close_j
                    exit_reason = "timeout"

            trades.append(Trade(
                ticker      = ticker,
                market      = market,
                entry_date  = df.index[entry_idx].to_pydatetime() if hasattr(df.index[entry_idx], "to_pydatetime") else df.index[entry_idx],
                exit_date   = df.index[exit_idx].to_pydatetime()  if hasattr(df.index[exit_idx],  "to_pydatetime") else df.index[exit_idx],
                entry_price = float(entry_price),
                exit_price  = float(exit_price),
                stop_loss   = float(stop_loss),
                exit_reason = exit_reason,
                regime      = regime,
                sector      = sector,
            ))

            # Passe après la fin du trade
            i = exit_idx + 1
        else:
            i += 1

    return trades


def _compute_stats(market: str, trades: list[Trade]) -> MarketStats:
    if not trades:
        return MarketStats(market=market)

    returns = [t.pnl_pct for t in trades]
    wins    = [r for r in returns if r > 0]

    # Drawdown : simuler un compte cumulatif
    cumulative = 0.0
    peak       = 0.0
    max_dd     = 0.0
    for r in returns:
        cumulative += r
        peak = max(peak, cumulative)
        dd   = peak - cumulative
        max_dd = max(max_dd, dd)

    # Sharpe (annualisé, rf=0)
    avg_r = float(np.mean(returns))
    std_r = float(np.std(returns)) if len(returns) > 1 else 1.0
    sharpe = (avg_r / std_r * math.sqrt(252 / MAX_HOLD)) if std_r > 0 else 0.0

    hold_days = [
        (t.exit_date - t.entry_date).days if hasattr(t.exit_date - t.entry_date, "days") else MAX_HOLD
        for t in trades
    ]

    return MarketStats(
        market        = market,
        n_trades      = len(trades),
        win_rate      = round(len(wins) / len(trades) * 100, 1),
        avg_return    = round(avg_r, 2),
        total_return  = round(sum(returns), 2),
        max_drawdown  = round(max_dd, 2),
        sharpe        = round(sharpe, 2),
        best_trade    = round(max(returns), 2),
        worst_trade   = round(min(returns), 2),
        avg_hold_days = round(float(np.mean(hold_days)), 1),
        trades        = trades,
    )


def run_backtest(db: Session) -> dict[str, MarketStats]:
    """Lance le backtest sur tous les marchés. Retourne stats par marché."""
    all_trades: dict[str, list[Trade]] = {}

    stocks = db.query(Stock).filter(Stock.is_active.is_(True)).all()
    logger.info(f"Backtest sur {len(stocks)} actions...")

    for stock in stocks:
        df = get_dataframe(db, stock)
        if df.empty:
            continue
        try:
            fund_score = (
                db.query(AnalysisResult.fundamental_score)
                .filter(
                    AnalysisResult.stock_id == stock.id,
                    AnalysisResult.fundamental_score.isnot(None),
                )
                .order_by(AnalysisResult.date.desc())
                .limit(1)
                .scalar()
            )
            trades = _run_stock_backtest(df, stock.ticker, stock.market, fundamental_score=fund_score, sector=stock.sector or "")
            all_trades.setdefault(stock.market, []).extend(trades)
        except Exception as e:
            logger.warning(f"[{stock.ticker}] backtest failed: {e}")

    results = {}
    for market, trades in all_trades.items():
        results[market] = _compute_stats(market, trades)
        logger.info(
            f"{market}: {results[market].n_trades} trades | "
            f"Win {results[market].win_rate}% | "
            f"Avg {results[market].avg_return:+.2f}%"
        )

    # Stats globales toutes places confondues
    all_t = [t for tlist in all_trades.values() for t in tlist]
    results["GLOBAL"] = _compute_stats("GLOBAL", all_t)

    return results


def stats_to_dict(stats: MarketStats) -> dict:
    """Sérialise pour template Jinja2."""
    top5_wins  = sorted(stats.trades, key=lambda t: t.pnl_pct, reverse=True)[:5]
    top5_loss  = sorted(stats.trades, key=lambda t: t.pnl_pct)[:5]

    # Breakdown par raison de sortie
    exit_counts: dict[str, int] = {}
    for t in stats.trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    # Breakdown par régime : win_rate + avg_return
    regime_stats: dict[str, dict] = {}
    for regime in ("trend_bull", "trend_bear", "range_bull", "range_bear"):
        rtrades = [t for t in stats.trades if t.regime == regime]
        if rtrades:
            rets  = [t.pnl_pct for t in rtrades]
            wins  = [r for r in rets if r > 0]
            regime_stats[regime] = {
                "n":          len(rtrades),
                "win_rate":   round(len(wins) / len(rtrades) * 100, 1),
                "avg_return": round(float(np.mean(rets)), 2),
            }

    # Breakdown par secteur
    _by_sector: dict[str, list] = {}
    for t in stats.trades:
        key = t.sector if t.sector else "—"
        _by_sector.setdefault(key, []).append(t)
    sector_stats: dict[str, dict] = {}
    for s, strades in _by_sector.items():
        rets = [t.pnl_pct for t in strades]
        wins = [r for r in rets if r > 0]
        sector_stats[s] = {
            "n":          len(strades),
            "win_rate":   round(len(wins) / len(strades) * 100, 1),
            "avg_return": round(float(np.mean(rets)), 2),
        }
    # Trier par retour moyen décroissant
    sector_stats = dict(sorted(sector_stats.items(), key=lambda x: x[1]["avg_return"], reverse=True))

    return {
        "market":        stats.market,
        "n_trades":      stats.n_trades,
        "win_rate":      stats.win_rate,
        "avg_return":    stats.avg_return,
        "total_return":  stats.total_return,
        "max_drawdown":  stats.max_drawdown,
        "sharpe":        stats.sharpe,
        "best_trade":    stats.best_trade,
        "worst_trade":   stats.worst_trade,
        "avg_hold_days": stats.avg_hold_days,
        "exit_counts":   exit_counts,
        "regime_stats":  regime_stats,
        "sector_stats":  sector_stats,
        "top_wins": [{"ticker": t.ticker, "pnl": round(t.pnl_pct, 2),
                      "entry": t.entry_date.strftime("%d/%m/%y"),
                      "exit": t.exit_date.strftime("%d/%m/%y"),
                      "reason": t.exit_reason} for t in top5_wins],
        "top_loss": [{"ticker": t.ticker, "pnl": round(t.pnl_pct, 2),
                      "entry": t.entry_date.strftime("%d/%m/%y"),
                      "exit": t.exit_date.strftime("%d/%m/%y"),
                      "reason": t.exit_reason} for t in top5_loss],
    }
