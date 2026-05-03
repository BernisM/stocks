import json
import logging
import os
import threading
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .auth import _decode_token, get_current_user
from .database import init_db
from .events import log_event
from .models import User
from .routers import analyse_router, auth_router, backtest_router, dashboard, guide_router, monitor_router, news_router, portfolio, recipients_router, stocks_router, watchlist_router
from .scheduler import job_update_and_score, start_scheduler

# ── Verrou global : une seule opération RAM-lourde à la fois ──────────────────
_HEAVY_LOCK = threading.Lock()

_JOB_TIMES: dict = {
    "sync_prices": None,
    "train_ml":    None,
    "fondamentaux": None,
}

_JOB_TIMINGS_PATH = "./ml_models/job_timings.json"
_MAX_JOB_LOG = 150

_JOB_LABELS = {
    "sync_prices":      "🔄 Sync Prix",
    "sync_fast":        "⚡ Sync Fast",
    "run_now":          "🗂 Run complet",
    "train_ml":         "🤖 Train ML",
    "fondamentaux":     "📊 Fondamentaux",
    "email_morning":    "📧 Email matin",
    "email_afternoon":  "📧 Email après-midi",
    "morning_chain":    "⛓ Chaîne matin",
    "afternoon_chain":  "⛓ Chaîne après-midi",
}


def _append_job_timing(job: str, started: datetime, finished: datetime, status: str, error: str | None = None) -> None:
    duration_s = round((finished - started).total_seconds())
    entry = {
        "job":        job,
        "label":      _JOB_LABELS.get(job, job),
        "started":    started.isoformat(),
        "finished":   finished.isoformat(),
        "duration_s": duration_s,
        "status":     status,
        "error":      error,
    }
    try:
        timings: list = []
        if os.path.exists(_JOB_TIMINGS_PATH):
            with open(_JOB_TIMINGS_PATH) as f:
                timings = json.load(f)
        timings.append(entry)
        timings = timings[-_MAX_JOB_LOG:]
        os.makedirs(os.path.dirname(_JOB_TIMINGS_PATH), exist_ok=True)
        with open(_JOB_TIMINGS_PATH, "w") as f:
            json.dump(timings, f)
    except Exception as exc:
        logger.warning(f"[job_timings] save failed: {exc}")
    m, s = divmod(duration_s, 60)
    dur_str = f"{m}m{s:02d}s" if m else f"{s}s"
    logger.info(f"[timing] {entry['label']} — {status.upper()} — {dur_str}"
                + (f" — {error}" if error else ""))

_CHAIN_STATE: dict = {
    "running":  False,
    "session":  "",        # "morning" | "afternoon"
    "step":     "",        # "run_now" | "sync_fast" | "train_ml" | "send_email" | "done"
    "started":  None,
    "finished": None,
    "error":    None,
}

_SMART_STATE: dict = {
    "running":    False,
    "jobs_done":  [],
    "email_sent": False,
    "error":      None,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

import yfinance as _yf
_yf.set_tz_cache_location("/app/cache/yfinance")

app = FastAPI(title="StockAnalyzer", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth_router.router)
app.include_router(dashboard.router)
app.include_router(portfolio.router)
app.include_router(backtest_router.router)
app.include_router(guide_router.router)
app.include_router(recipients_router.router)
app.include_router(stocks_router.router)
app.include_router(analyse_router.router)
app.include_router(monitor_router.router)
app.include_router(news_router.router)
app.include_router(watchlist_router.router)


_LOG_GET_PATHS  = frozenset({"/dashboard", "/portfolio", "/analyse", "/backtest", "/guide"})
_LOG_POST_PATHS = frozenset({"/api/analyse/run", "/sync-prices", "/sync-tickers"})


# Chemins accessibles sans être connecté
_PUBLIC_PREFIXES = (
    "/login", "/register", "/logout",
    "/static/",
    "/ping",
    "/admin/",
    "/admin-panel",
    "/favicon.ico", "/apple-touch-icon",
)

@app.middleware("http")
async def _auth_guard(request: Request, call_next):
    path = request.url.path
    if not any(path == p or path.startswith(p) for p in _PUBLIC_PREFIXES):
        token = request.cookies.get("access_token")
        if not token or _decode_token(token) is None:
            return RedirectResponse("/login", status_code=302)
    return await call_next(request)


@app.middleware("http")
async def _request_logger(request: Request, call_next):
    response = await call_next(request)
    path   = request.url.path
    method = request.method
    if (method == "GET" and path in _LOG_GET_PATHS) or \
       (method == "POST" and path in _LOG_POST_PATHS):
        token = request.cookies.get("access_token")
        ip    = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
                or (request.client.host if request.client else "")
        def _write(tok=token, p=path, remote_ip=ip):
            from .database import SessionLocal
            from .models import User, UserEvent
            db = SessionLocal()
            try:
                uid   = _decode_token(tok) if tok else None
                email = None
                if uid:
                    u = db.query(User).filter(User.id == uid).first()
                    email = u.email if u else None
                db.add(UserEvent(user_email=email, event_type="page" if method == "GET" else "action",
                                 detail=p, ip=remote_ip))
                db.commit()
            except Exception:
                pass
            finally:
                db.close()
        threading.Thread(target=_write, daemon=True).start()
    return response


@app.on_event("startup")
def on_startup():
    init_db()
    start_scheduler()


@app.get("/")
def root():
    return RedirectResponse("/login")


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/admin-panel")
def admin_panel():
    from fastapi.responses import HTMLResponse
    with open("templates/admin_panel.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── Sync prix en temps réel ────────────────────────────────────────────────────

_sync_state: dict = {
    "running":  False,
    "phase":    "",      # "prices" | "scores" | "done"
    "progress": 0,
    "total":    0,
    "started":  None,
    "finished": None,
    "error":    None,
}


_ticker_sync_state: dict = {
    "running": False,
    "done":    False,
    "count":   0,
    "error":   None,
}


@app.post("/sync-tickers")
async def sync_tickers(request: Request, user: User = Depends(get_current_user)):
    body    = await request.json()
    tickers = [t.strip().upper() for t in body.get("tickers", []) if t.strip()]
    if not tickers:
        return JSONResponse({"status": "error", "message": "Aucun ticker fourni"}, status_code=400)

    def _run():
        if not _HEAVY_LOCK.acquire(blocking=False):
            _ticker_sync_state.update({"running": False, "done": False, "error": "Une opération lourde est déjà en cours, réessayez dans quelques minutes."})
            return
        from .data_engine import sync_selected_tickers
        from .database import SessionLocal
        _ticker_sync_state.update({"running": True, "done": False, "count": len(tickers), "error": None})
        db = SessionLocal()
        try:
            sync_selected_tickers(db, tickers)
            _ticker_sync_state["done"] = True
        except Exception as e:
            _ticker_sync_state["error"] = str(e)
        finally:
            _ticker_sync_state["running"] = False
            db.close()
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "tickers": tickers, "count": len(tickers)})


@app.get("/ticker-sync-status")
def ticker_sync_status(user: User = Depends(get_current_user)):
    return JSONResponse(_ticker_sync_state)


@app.post("/sync-prices")
def sync_prices(user: User = Depends(get_current_user)):
    if _sync_state["running"]:
        return JSONResponse({"status": "already_running"})
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "already_running", "message": "Une opération lourde est déjà en cours."})

    def _run():
        from .data_engine import sync_prices_fast
        from .database import SessionLocal
        db = SessionLocal()
        t0 = datetime.now(UTC).replace(tzinfo=None)
        _sync_state.update({
            "running": True, "phase": "prices",
            "progress": 0, "total": 0,
            "started": t0.isoformat(), "error": None,
        })
        try:
            def on_progress(done, total, phase):
                _sync_state["phase"]    = phase
                _sync_state["progress"] = done
                _sync_state["total"]    = total

            sync_prices_fast(db, on_progress=on_progress)
            t1 = datetime.now(UTC).replace(tzinfo=None)
            _sync_state["phase"]    = "done"
            _sync_state["finished"] = t1.isoformat()
            _JOB_TIMES["sync_prices"] = t1.isoformat()
            _append_job_timing("sync_prices", t0, t1, "ok")
        except Exception as e:
            t1 = datetime.now(UTC).replace(tzinfo=None)
            _sync_state["error"] = str(e)
            _append_job_timing("sync_prices", t0, t1, "error", str(e))
        finally:
            _sync_state["running"] = False
            db.close()
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.get("/job-status")
def job_status(user: User = Depends(get_current_user)):
    return JSONResponse(_JOB_TIMES)


@app.get("/smart-email-status")
def smart_email_status(user: User = Depends(get_current_user)):
    return JSONResponse(_SMART_STATE)


@app.post("/send-email-smart")
async def send_email_smart(request: Request, user: User = Depends(get_current_user)):
    body = await request.json()
    jobs = body.get("jobs", [])

    ETA = {"sync_prices": 8, "train_ml": 5, "fondamentaux": 6}
    eta_min = sum(ETA.get(j, 0) for j in jobs) + 1

    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})

    _SMART_STATE.update({"running": True, "jobs_done": [], "email_sent": False, "error": None})

    def _run():
        from .database import SessionLocal
        try:
            if "sync_prices" in jobs:
                from .data_engine import sync_prices_fast
                db = SessionLocal()
                try:
                    sync_prices_fast(db)
                    _JOB_TIMES["sync_prices"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
                    _SMART_STATE["jobs_done"].append("sync_prices")
                finally:
                    db.close()

            if "train_ml" in jobs:
                from .scheduler import job_retrain_ml
                job_retrain_ml()
                _JOB_TIMES["train_ml"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
                _SMART_STATE["jobs_done"].append("train_ml")

            if "fondamentaux" in jobs:
                from .fundamentals import update_fundamentals
                db = SessionLocal()
                try:
                    update_fundamentals(db)
                    _JOB_TIMES["fondamentaux"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
                    _SMART_STATE["jobs_done"].append("fondamentaux")
                finally:
                    db.close()

            # Send email
            from .data_engine import load_market_status
            from .email_sender import send_combined_report
            from .ml_model import load_metrics
            from .models import AnalysisResult, ExtraRecipient, Stock
            db = SessionLocal()
            try:
                last_date = db.query(AnalysisResult.date).order_by(AnalysisResult.date.desc()).limit(1).scalar()
                if last_date:
                    def get_top(market):
                        rows = (
                            db.query(AnalysisResult, Stock)
                            .join(Stock, AnalysisResult.stock_id == Stock.id)
                            .filter(AnalysisResult.date == last_date, Stock.market == market,
                                    AnalysisResult.ranking.in_(["Strong Buy", "Buy"]))
                            .order_by(AnalysisResult.score_final.desc()).limit(10).all()
                        )
                        return [{"ticker": s.ticker, "name": s.name or "", "close": ar.close or 0,
                                 "score_final": ar.score_final or 0, "ranking": ar.ranking or "Neutral",
                                 "stop_loss": ar.stop_loss_price or 0, "rsi": ar.rsi or 0,
                                 "macd_hist": ar.macd_hist or 0, "volatility": ar.volatility or 0,
                                 "ml_probability": ar.ml_probability,
                                 "atr_pct": (ar.atr / ar.close * 100) if ar.atr and ar.close else 0,
                                 "bollinger_b": ar.bollinger_b or 0} for ar, s in rows]
                    extras = db.query(ExtraRecipient).filter(ExtraRecipient.is_active.is_(True)).all()
                    recipients = [(user.email, user.level)] + [(e.email, e.level) for e in extras]
                    send_combined_report(recipients=recipients, top_cac40=get_top("CAC40"),
                                         top_sbf120=get_top("SBF120"), top_sp500=get_top("SP500"),
                                         analysis_date=last_date, ml_metrics=load_metrics(),
                                         top_commodities=get_top("COMMODITIES"),
                                         top_crypto=get_top("CRYPTO"),
                                         market_status=load_market_status())
                    _SMART_STATE["email_sent"] = True
            finally:
                db.close()
        except Exception as e:
            _SMART_STATE["error"] = str(e)
        finally:
            _SMART_STATE["running"] = False
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "eta_min": eta_min, "jobs": jobs})


@app.get("/sync-status")
def sync_status(user: User = Depends(get_current_user)):
    pct = 0
    if _sync_state["total"] > 0:
        pct = round(_sync_state["progress"] / _sync_state["total"] * 100)
    return JSONResponse({**_sync_state, "pct": pct})


# ── Envoi email à l'utilisateur connecté ──────────────────────────────────────

@app.get("/send-email-me")
def send_email_me(user: User = Depends(get_current_user)):
    def _run():
        from .data_engine import load_market_status
        from .database import SessionLocal
        from .email_sender import send_combined_report
        from .ml_model import load_metrics
        from .models import AnalysisResult, ExtraRecipient, Stock
        db = SessionLocal()
        try:
            last_date = (
                db.query(AnalysisResult.date)
                .order_by(AnalysisResult.date.desc())
                .limit(1)
                .scalar()
            )
            if not last_date:
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
                    .limit(10)
                    .all()
                )
                return [{
                    "ticker":         s.ticker,
                    "name":           s.name or "",
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
                } for ar, s in rows]

            extras = db.query(ExtraRecipient).filter(ExtraRecipient.is_active.is_(True)).all()
            recipients = [(user.email, user.level)] + [(e.email, e.level) for e in extras]
            send_combined_report(
                recipients=recipients,
                top_cac40=get_top("CAC40"),
                top_sbf120=get_top("SBF120"),
                top_sp500=get_top("SP500"),
                analysis_date=last_date,
                ml_metrics=load_metrics(),
                top_commodities=get_top("COMMODITIES"),
                top_crypto=get_top("CRYPTO"),
                market_status=load_market_status(),
            )
        finally:
            db.close()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": f"Email envoyé à {user.email}"})


# ── Admin endpoints ────────────────────────────────────────────────────────────

_backtest_state: dict = {
    "running":  False,
    "started":  None,
    "finished": None,
    "error":    None,
}


@app.post("/admin/backtest-run")
def backtest_run():
    if _backtest_state["running"]:
        return JSONResponse({"status": "already_running"})
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})

    def _run():
        import json
        from .backtest import run_backtest, stats_to_dict
        from .database import SessionLocal
        _backtest_state.update({
            "running": True,
            "started": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "finished": None, "error": None,
        })
        db = SessionLocal()
        try:
            results = run_backtest(db)
            cache = {k: stats_to_dict(v) for k, v in results.items()}
            with open("./ml_models/backtest_cache.json", "w") as f:
                json.dump(cache, f)
            _backtest_state["finished"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
        except Exception as e:
            _backtest_state["error"] = str(e)
        finally:
            _backtest_state["running"] = False
            db.close()
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.get("/backtest-status")
def backtest_status(user: User = Depends(get_current_user)):
    return JSONResponse(_backtest_state)


@app.get("/admin/fundamentals-now")
def fundamentals_now():
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})
    def _run():
        from .fundamentals import update_fundamentals
        from .database import SessionLocal
        db = SessionLocal()
        t0 = datetime.now(UTC).replace(tzinfo=None)
        try:
            update_fundamentals(db)
            t1 = datetime.now(UTC).replace(tzinfo=None)
            _JOB_TIMES["fondamentaux"] = t1.isoformat()
            _append_job_timing("fondamentaux", t0, t1, "ok")
        except Exception as e:
            _append_job_timing("fondamentaux", t0, datetime.now(UTC).replace(tzinfo=None), "error", str(e))
        finally:
            db.close()
            _HEAVY_LOCK.release()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Fondamentaux en cours (~6 min). Revenez dans quelques minutes."})


@app.get("/admin/train-ml")
def train_ml():
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})
    def _run():
        t0 = datetime.now(UTC).replace(tzinfo=None)
        try:
            from .scheduler import job_retrain_ml
            job_retrain_ml()
            t1 = datetime.now(UTC).replace(tzinfo=None)
            _JOB_TIMES["train_ml"] = t1.isoformat()
            _append_job_timing("train_ml", t0, t1, "ok")
        except Exception as e:
            _append_job_timing("train_ml", t0, datetime.now(UTC).replace(tzinfo=None), "error", str(e))
        finally:
            _HEAVY_LOCK.release()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Entraînement ML lancé (~5 min)."})


@app.get("/admin/send-email")
def send_email_now():
    def _run():
        from .scheduler import job_email_daily
        t0 = datetime.now(UTC).replace(tzinfo=None)
        try:
            job_email_daily()
            _append_job_timing("email_morning", t0, datetime.now(UTC).replace(tzinfo=None), "ok")
        except Exception as e:
            _append_job_timing("email_morning", t0, datetime.now(UTC).replace(tzinfo=None), "error", str(e))
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Email matin en cours d'envoi."})


@app.get("/admin/send-email-afternoon")
def send_email_afternoon():
    def _run():
        from .scheduler import job_email_afternoon
        t0 = datetime.now(UTC).replace(tzinfo=None)
        try:
            job_email_afternoon()
            _append_job_timing("email_afternoon", t0, datetime.now(UTC).replace(tzinfo=None), "ok")
        except Exception as e:
            _append_job_timing("email_afternoon", t0, datetime.now(UTC).replace(tzinfo=None), "error", str(e))
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Email après-midi en cours d'envoi."})


@app.get("/admin/refresh-tickers")
def refresh_tickers_endpoint():
    """
    Refresh hebdomadaire des listes de tickers (CAC40, SBF120, NASDAQ_GROWTH).
    Sources : Wikipedia + iShares IWO. Soft-delete pour les disparus.
    Envoie un email récap à EMAIL_USER si des changements sont détectés.
    """
    def _run():
        from .config import EMAIL_USER
        from .database import SessionLocal
        from .email_sender import send_ticker_diff_alert
        from .ticker_refresh import apply_diffs_to_db, refresh_all_dynamic

        diffs = refresh_all_dynamic()

        db = SessionLocal()
        try:
            db_diff = apply_diffs_to_db(db, diffs)
            logger.info(f"[refresh_tickers] DB sync : {db_diff}")
        finally:
            db.close()

        if EMAIL_USER:
            try:
                send_ticker_diff_alert(EMAIL_USER, diffs)
            except Exception as e:
                logger.warning(f"[refresh_tickers] email diff failed: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({
        "status": "started",
        "message": "Refresh tickers lancé (~30s). Email envoyé si changements détectés.",
    })


@app.get("/admin/ticker-cache")
def ticker_cache_status():
    """État actuel du cache de refresh tickers (lecture seule)."""
    from .ticker_refresh import _load_cache
    cache = _load_cache()
    return JSONResponse({
        market: {
            "count":        entry.get("count", 0),
            "last_refresh": entry.get("last_refresh"),
        }
        for market, entry in cache.items()
        if isinstance(entry, dict)
    })


@app.get("/admin/sync-fast")
def sync_fast():
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})
    def _run():
        from .data_engine import sync_prices_fast
        from .database import SessionLocal
        db = SessionLocal()
        t0 = datetime.now(UTC).replace(tzinfo=None)
        try:
            sync_prices_fast(db)
            _append_job_timing("sync_fast", t0, datetime.now(UTC).replace(tzinfo=None), "ok")
        except Exception as e:
            _append_job_timing("sync_fast", t0, datetime.now(UTC).replace(tzinfo=None), "error", str(e))
        finally:
            db.close()
            _HEAVY_LOCK.release()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Sync rapide lancé (~8 min)."})


@app.get("/admin/run-now")
def run_now():
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})
    def _run():
        t0 = datetime.now(UTC).replace(tzinfo=None)
        try:
            job_update_and_score()
            _append_job_timing("run_now", t0, datetime.now(UTC).replace(tzinfo=None), "ok")
        except Exception as e:
            _append_job_timing("run_now", t0, datetime.now(UTC).replace(tzinfo=None), "error", str(e))
        finally:
            _HEAVY_LOCK.release()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Téléchargement complet lancé (~30-60 min)."})


# ── Chaînes séquentielles (1 seul appel cron-job.org) ─────────────────────────

@app.get("/admin/morning-chain")
def morning_chain():
    """Enchaîne train-ml → run-now → send-email dans un seul thread.
    train-ml en premier : le scoring utilise le nouveau modèle → ML prob correctes."""
    # Cooldown : si la chaîne matin a tourné il y a moins de 3h, ignorer
    if _CHAIN_STATE.get("session") == "morning" and _CHAIN_STATE.get("finished"):
        try:
            finished_at = datetime.fromisoformat(_CHAIN_STATE["finished"])
            elapsed = (datetime.now(UTC).replace(tzinfo=None) - finished_at).total_seconds()
            if elapsed < 3 * 3600:
                return JSONResponse({"status": "skipped", "message": f"Chaîne matin déjà terminée il y a {int(elapsed//60)} min."})
        except Exception:
            pass

    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})

    def _run():
        from .scheduler import job_email_daily, job_retrain_ml
        chain_t0 = datetime.now(UTC).replace(tzinfo=None)
        _CHAIN_STATE.update({
            "running": True, "session": "morning", "step": "train_ml",
            "started": chain_t0.isoformat(),
            "finished": None, "error": None,
        })
        try:
            logging.getLogger(__name__).info("[morning-chain] Étape 1 : train-ml")
            t0 = datetime.now(UTC).replace(tzinfo=None)
            job_retrain_ml()
            t1 = datetime.now(UTC).replace(tzinfo=None)
            _JOB_TIMES["train_ml"] = t1.isoformat()
            _append_job_timing("train_ml", t0, t1, "ok")

            _CHAIN_STATE["step"] = "run_now"
            logging.getLogger(__name__).info("[morning-chain] Étape 2 : run-now")
            t0 = datetime.now(UTC).replace(tzinfo=None)
            job_update_and_score()
            t1 = datetime.now(UTC).replace(tzinfo=None)
            _JOB_TIMES["sync_prices"] = t1.isoformat()
            _append_job_timing("run_now", t0, t1, "ok")

            _CHAIN_STATE["step"] = "send_email"
            logging.getLogger(__name__).info("[morning-chain] Étape 3 : send-email")
            t0 = datetime.now(UTC).replace(tzinfo=None)
            job_email_daily()
            _append_job_timing("email_morning", t0, datetime.now(UTC).replace(tzinfo=None), "ok")

            chain_t1 = datetime.now(UTC).replace(tzinfo=None)
            _CHAIN_STATE["step"] = "done"
            _CHAIN_STATE["finished"] = chain_t1.isoformat()
            _append_job_timing("morning_chain", chain_t0, chain_t1, "ok")
            logging.getLogger(__name__).info("[morning-chain] Terminé ✅")
        except Exception as e:
            _CHAIN_STATE["error"] = str(e)
            _append_job_timing("morning_chain", chain_t0, datetime.now(UTC).replace(tzinfo=None), "error", str(e))
            logging.getLogger(__name__).error(f"[morning-chain] Erreur : {e}")
        finally:
            _CHAIN_STATE["running"] = False
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Chaîne matin lancée : train-ml → run-now → send-email."})


@app.get("/admin/afternoon-chain")
def afternoon_chain():
    """Enchaîne train-ml → sync-fast → send-email-afternoon dans un seul thread."""
    # Cooldown : si la chaîne après-midi a tourné il y a moins de 3h, ignorer
    if _CHAIN_STATE.get("session") == "afternoon" and _CHAIN_STATE.get("finished"):
        try:
            finished_at = datetime.fromisoformat(_CHAIN_STATE["finished"])
            elapsed = (datetime.now(UTC).replace(tzinfo=None) - finished_at).total_seconds()
            if elapsed < 3 * 3600:
                return JSONResponse({"status": "skipped", "message": f"Chaîne après-midi déjà terminée il y a {int(elapsed//60)} min."})
        except Exception:
            pass

    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})

    def _run():
        from .data_engine import sync_prices_fast
        from .database import SessionLocal
        from .scheduler import job_email_afternoon, job_retrain_ml
        chain_t0 = datetime.now(UTC).replace(tzinfo=None)
        _CHAIN_STATE.update({
            "running": True, "session": "afternoon", "step": "train_ml",
            "started": chain_t0.isoformat(),
            "finished": None, "error": None,
        })
        try:
            logging.getLogger(__name__).info("[afternoon-chain] Étape 1 : train-ml")
            t0 = datetime.now(UTC).replace(tzinfo=None)
            job_retrain_ml()
            t1 = datetime.now(UTC).replace(tzinfo=None)
            _JOB_TIMES["train_ml"] = t1.isoformat()
            _append_job_timing("train_ml", t0, t1, "ok")

            _CHAIN_STATE["step"] = "sync_fast"
            logging.getLogger(__name__).info("[afternoon-chain] Étape 2 : sync-fast")
            t0 = datetime.now(UTC).replace(tzinfo=None)
            db = SessionLocal()
            try:
                sync_prices_fast(db)
            finally:
                db.close()
            t1 = datetime.now(UTC).replace(tzinfo=None)
            _JOB_TIMES["sync_prices"] = t1.isoformat()
            _append_job_timing("sync_fast", t0, t1, "ok")

            _CHAIN_STATE["step"] = "send_email"
            logging.getLogger(__name__).info("[afternoon-chain] Étape 3 : send-email-afternoon")
            t0 = datetime.now(UTC).replace(tzinfo=None)
            job_email_afternoon()
            _append_job_timing("email_afternoon", t0, datetime.now(UTC).replace(tzinfo=None), "ok")

            chain_t1 = datetime.now(UTC).replace(tzinfo=None)
            _CHAIN_STATE["step"] = "done"
            _CHAIN_STATE["finished"] = chain_t1.isoformat()
            _append_job_timing("afternoon_chain", chain_t0, chain_t1, "ok")
            logging.getLogger(__name__).info("[afternoon-chain] Terminé ✅")
        except Exception as e:
            _CHAIN_STATE["error"] = str(e)
            _append_job_timing("afternoon_chain", chain_t0, datetime.now(UTC).replace(tzinfo=None), "error", str(e))
            logging.getLogger(__name__).error(f"[afternoon-chain] Erreur : {e}")
        finally:
            _CHAIN_STATE["running"] = False
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Chaîne après-midi lancée : train-ml → sync-fast → send-email-afternoon."})


@app.get("/admin/chain-status")
def chain_status():
    return JSONResponse(_CHAIN_STATE)


@app.get("/admin/job-timings")
def job_timings():
    try:
        if os.path.exists(_JOB_TIMINGS_PATH):
            with open(_JOB_TIMINGS_PATH) as f:
                return JSONResponse(list(reversed(json.load(f))))
    except Exception:
        pass
    return JSONResponse([])


# ── Admin : gestion des tickers ───────────────────────────────────────────────

@app.get("/admin/tickers")
def admin_tickers(
    request: Request,
    market: str = "",
    status: str = "active",
    q: str = "",
):
    from fastapi.templating import Jinja2Templates as _T
    from sqlalchemy import func, or_
    from .database import SessionLocal
    from .models import Stock

    _templates = _T(directory="templates")
    db = SessionLocal()
    try:
        query = db.query(Stock)
        if market:
            query = query.filter(Stock.market == market)
        if status == "active":
            query = query.filter(Stock.is_active.is_(True))
        elif status == "inactive":
            query = query.filter(Stock.is_active.is_(False))
        if q:
            pat = f"%{q.strip().lower()}%"
            query = query.filter(or_(
                func.lower(Stock.ticker).like(pat),
                func.lower(Stock.name).like(pat),
            ))
        stocks = query.order_by(Stock.market, Stock.ticker).all()
        total = db.query(Stock).count()
        total_active = db.query(Stock).filter(Stock.is_active.is_(True)).count()
        total_inactive = total - total_active
        markets = sorted({r[0] for r in db.query(Stock.market).distinct().all()})
    finally:
        db.close()

    # Try to get the logged-in user for the template (optional for admin pages)
    try:
        token = request.cookies.get("access_token")
        from .auth import _decode_token
        from .database import SessionLocal as SL
        db2 = SL()
        uid = _decode_token(token) if token else None
        user = db2.query(User).filter(User.id == uid).first() if uid else None
        db2.close()
    except Exception:
        user = None

    return _templates.TemplateResponse(request, "admin_tickers.html", {
        "user":           user,
        "stocks":         stocks,
        "markets":        markets,
        "sel_market":     market,
        "sel_status":     status,
        "sel_q":          q,
        "total":          total,
        "total_active":   total_active,
        "total_inactive": total_inactive,
    })


@app.post("/admin/tickers/toggle/{ticker}")
def admin_toggle_ticker(ticker: str):
    from .database import SessionLocal
    from .models import Stock
    import pytz
    db = SessionLocal()
    try:
        stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
        if not stock:
            return JSONResponse({"error": "ticker not found"}, status_code=404)
        stock.is_active = not stock.is_active
        if not stock.is_active:
            stock.delisted_at = datetime.now(pytz.utc).replace(tzinfo=None)
        else:
            stock.delisted_at = None
        db.commit()
        return JSONResponse({
            "is_active": stock.is_active,
            "message":   f"{ticker.upper()} {'réactivé' if stock.is_active else 'désactivé'}.",
        })
    finally:
        db.close()


# ── Blacklist tickers ──────────────────────────────────────────────────────────

@app.get("/admin/blacklisted-tickers")
def admin_blacklisted_tickers():
    from .data_engine import get_blacklisted
    return JSONResponse(get_blacklisted())


@app.post("/admin/unblacklist/{ticker:path}")
def admin_unblacklist(ticker: str):
    from .data_engine import unblacklist
    unblacklist(ticker)
    return JSONResponse({"status": "ok", "ticker": ticker})


# ── Recalcul hyper-growth (sans appel yfinance) ───────────────────────────────

@app.get("/admin/recalc-hyper-growth")
def admin_recalc_hyper_growth():
    """Recalcule hyper_growth_score pour tous les stocks éligibles (no yfinance, ~2-3 min)."""
    from .models import AnalysisResult, Stock

    def _run():
        from sqlalchemy import text
        from .data_engine import get_dataframe
        from .indicators import compute_indicators, get_last_row
        from .scoring import compute_hyper_growth_score
        from .database import SessionLocal

        db = SessionLocal()
        updated = errors = skipped = 0
        try:
            # Pré-filtre SQL : seuls les stocks avec rev_growth >= 15 et score_final >= 55
            rows = db.execute(text("""
                SELECT DISTINCT ON (s.id) s.id, ar.id AS ar_id
                FROM stocks s
                JOIN analysis_results ar ON ar.stock_id = s.id
                WHERE s.is_active = TRUE
                  AND s.market NOT IN ('COMMODITIES', 'CRYPTO')
                  AND ar.fundamental_score IS NOT NULL
                  AND ar.rev_growth >= 15
                  AND (ar.debt_equity IS NULL OR ar.debt_equity <= 300)
                  AND ar.score_final >= 55
                ORDER BY s.id, ar.date DESC
            """)).fetchall()
            logger.info(f"[recalc-hyper-growth] {len(rows)} stocks pré-filtrés — chargement indicateurs…")

            stock_map = {s.id: s for s in db.query(Stock).filter(Stock.is_active.is_(True)).all()}

            for row in rows:
                stock = stock_map.get(row[0])
                ar = db.get(AnalysisResult, row[1])
                if not stock or not ar:
                    continue
                try:
                    df = get_dataframe(db, stock)
                    if df.empty or len(df) < 30:
                        skipped += 1
                        continue
                    df = compute_indicators(df)
                    ind = get_last_row(df)
                    ar.hyper_growth_score = compute_hyper_growth_score(
                        ind,
                        ar.rev_growth,
                        ar.debt_equity,
                        ar.fundamental_score,
                        ar.score_final or 0,
                        ar.fcf,
                        ar.pb_ratio,
                    )
                    db.commit()
                    updated += 1
                except Exception as e:
                    db.rollback()
                    errors += 1
                    logger.warning(f"[recalc-hg] {stock.ticker}: {e}")
        finally:
            db.close()
        logger.info(f"[recalc-hyper-growth] {updated} mis à jour, {skipped} ignorés, {errors} erreurs")

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started"})


# ── Mise à jour sentiment news (Buy + Strong Buy uniquement) ──────────────────

@app.get("/admin/update-news-sentiment")
def admin_update_news_sentiment():
    """Calcule news_sentiment pour les stocks Buy/Strong Buy (~3-4 min, 0.3s/ticker)."""
    from .news import update_news_sentiment_for_signals

    def _run():
        from .database import SessionLocal as _SL
        db = _SL()
        try:
            result = update_news_sentiment_for_signals(db)
            logger.info(f"[update-news-sentiment] {result}")
        except Exception as e:
            logger.error(f"[update-news-sentiment] {e}")
        finally:
            db.close()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started"})
