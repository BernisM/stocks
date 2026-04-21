"""
Planificateur APScheduler :
  02h00 (lun-ven) : mise à jour données + scoring
  02h00 (dim)     : + ré-entraînement ML
  09h05 (lun-ven) : email Top 10 Europe
  15h35 (lun-ven) : email Top 10 US
  toutes les 15 min (heures de marché) : vérification stop-loss portefeuilles
"""
import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import TOP_N_EMAIL
from .database import SessionLocal
from .models import AnalysisResult, ExtraRecipient, PortfolioPosition, Stock, User

logger = logging.getLogger(__name__)
TZ = pytz.timezone("Europe/Paris")


# ── Job : mise à jour + scoring ───────────────────────────────────────────────

def job_update_and_score():
    from .data_engine import get_dataframe, update_all_markets
    from .indicators import compute_indicators, get_last_row
    from .ml_model import predict
    from .scoring import compute_score

    logger.info("=== job_update_and_score démarré ===")
    db = SessionLocal()
    try:
        update_all_markets(db)
        stocks = db.query(Stock).all()
        today  = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        for stock in stocks:
            try:
                df = get_dataframe(db, stock)
                if df.empty or len(df) < 30:
                    continue
                df = compute_indicators(df)
                ind = get_last_row(df)
                ml_prob = predict(df)
                score_base, ml_boost, score_final, ranking = compute_score(ind, ml_prob)

                existing = (
                    db.query(AnalysisResult)
                    .filter(AnalysisResult.stock_id == stock.id, AnalysisResult.date == today)
                    .first()
                )
                if not existing:
                    existing = AnalysisResult(stock_id=stock.id, date=today)
                    db.add(existing)

                existing.close           = ind.get("Close")
                existing.atr             = ind.get("ATR")
                existing.stop_loss_price = ind.get("Stop_Loss")
                existing.volatility      = ind.get("Volatility")
                existing.rsi             = ind.get("RSI")
                existing.macd            = ind.get("MACD")
                existing.macd_signal     = ind.get("MACD_signal")
                existing.macd_hist       = ind.get("MACD_hist")
                existing.bollinger_b     = ind.get("BB_pct")
                existing.ema50           = ind.get("EMA50")
                existing.sma200          = ind.get("SMA200")
                existing.volume_ratio    = ind.get("Vol_ratio")
                existing.score_base      = score_base
                existing.ml_probability  = ml_prob
                existing.ml_boost        = ml_boost
                existing.score_final     = score_final
                existing.ranking         = ranking

                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning(f"[{stock.ticker}] scoring failed: {e}")

    finally:
        db.close()
    logger.info("=== job_update_and_score terminé ===")


# ── Job : fondamentaux (dimanche) ────────────────────────────────────────────

def job_update_fundamentals():
    from .fundamentals import update_fundamentals
    logger.info("=== job_update_fundamentals démarré ===")
    db = SessionLocal()
    try:
        update_fundamentals(db)
    finally:
        db.close()
    logger.info("=== job_update_fundamentals terminé ===")


# ── Job : ré-entraînement ML (dimanche) ──────────────────────────────────────

def job_retrain_ml():
    from .data_engine import get_dataframe
    from .indicators import compute_indicators
    from .ml_model import save_metrics, train

    logger.info("=== job_retrain_ml démarré ===")
    db = SessionLocal()
    try:
        stocks = db.query(Stock).all()
        dfs = []
        for stock in stocks:
            df = get_dataframe(db, stock)
            if not df.empty and len(df) >= 60:
                df = compute_indicators(df)
                dfs.append(df)
        metrics = train(dfs)
        if metrics:
            save_metrics(metrics)
    finally:
        db.close()
    logger.info("=== job_retrain_ml terminé ===")


# ── Job : email quotidien (Europe + US en un seul email) ─────────────────────

def job_email_daily():
    from .email_sender import send_combined_report
    from .ml_model import load_metrics

    db = SessionLocal()
    try:
        last_date = (
            db.query(AnalysisResult.date)
            .order_by(AnalysisResult.date.desc())
            .limit(1)
            .scalar()
        )
        if not last_date:
            logger.info("Aucune donnée pour l'email")
            return

        def get_top(market):
            rows = (
                db.query(AnalysisResult, Stock)
                .join(Stock, AnalysisResult.stock_id == Stock.id)
                .filter(
                    AnalysisResult.date == last_date,
                    Stock.market == market,
                    AnalysisResult.ranking.in_(["Strong Buy", "Buy"]),
                )
                .order_by(AnalysisResult.score_final.desc())
                .limit(TOP_N_EMAIL)
                .all()
            )
            return [{
                "ticker":         stock.ticker,
                "name":           stock.name or "",
                "close":          ar.close or 0,
                "score_final":    ar.score_final or 0,
                "ranking":        ar.ranking or "Neutral",
                "stop_loss":      ar.stop_loss_price or 0,
                "rsi":            ar.rsi or 0,
                "macd_hist":      ar.macd_hist or 0,
                "volatility":     ar.volatility or 0,
                "ml_probability": ar.ml_probability,
                "atr_pct":        (ar.atr / ar.close * 100) if ar.atr and ar.close else 0,
                "bollinger_b":    ar.bollinger_b or 0,
            } for ar, stock in rows]

        users      = db.query(User).filter(User.is_active == True).all()
        extras     = db.query(ExtraRecipient).all()
        recipients = [(u.email, u.level) for u in users] + [(e.email, e.level) for e in extras]
        ml_metrics = load_metrics()

        send_combined_report(
            recipients=recipients,
            top_cac40=get_top("CAC40"),
            top_sbf120=get_top("SBF120"),
            top_sp500=get_top("SP500"),
            analysis_date=last_date,
            ml_metrics=ml_metrics or None,
        )

    finally:
        db.close()


# ── Job : vérification stop-loss portefeuilles ────────────────────────────────

def job_check_stop_losses():
    from .data_engine import get_current_price
    from .email_sender import send_stop_loss_alert

    db = SessionLocal()
    try:
        positions = (
            db.query(PortfolioPosition)
            .filter(PortfolioPosition.is_active == True,
                    PortfolioPosition.stop_loss_price != None)
            .all()
        )
        for pos in positions:
            price = get_current_price(pos.ticker)
            if price is None:
                continue
            if price <= pos.stop_loss_price:
                user = db.query(User).filter(User.id == pos.user_id).first()
                if user:
                    send_stop_loss_alert(
                        email=user.email,
                        ticker=pos.ticker,
                        current_price=price,
                        stop_price=pos.stop_loss_price,
                        buy_price=pos.buy_price,
                        level=user.level,
                    )
                    logger.info(f"Stop-loss alert sent: {pos.ticker} @ {price}")
    finally:
        db.close()


# ── Démarrage du scheduler ────────────────────────────────────────────────────

def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=TZ)

    # Mise à jour données + scoring : lun-sam à 2h00
    scheduler.add_job(
        job_update_and_score,
        CronTrigger(day_of_week="mon-sat", hour=2, minute=0, timezone=TZ),
        id="update_score", replace_existing=True,
    )

    # Ré-entraînement ML : dimanche à 2h00
    scheduler.add_job(
        job_retrain_ml,
        CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=TZ),
        id="retrain_ml", replace_existing=True,
    )

    # Fondamentaux : dimanche à 3h00 (après ML, ~6 min pour 667 stocks)
    scheduler.add_job(
        job_update_fundamentals,
        CronTrigger(day_of_week="sun", hour=3, minute=0, timezone=TZ),
        id="fundamentals", replace_existing=True,
    )

    # Stop-loss : toutes les 15 min, lun-ven 9h-22h
    scheduler.add_job(
        job_check_stop_losses,
        CronTrigger(day_of_week="mon-fri", hour="9-21", minute="*/15", timezone=TZ),
        id="stop_loss", replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler démarré.")
    return scheduler
