# StockAnalyzer — CLAUDE.md

## Project overview

Full-stack stock analysis platform built with FastAPI + SQLite.
Analyzes ~667 stocks across CAC40, SBF120, SP500, NASDAQ with:
- Technical indicators (RSI, MACD, ATR, Bollinger Bands, OBV, Ichimoku)
- RandomForest ML scoring (~66.8% accuracy, 100k+ observations)
- Fundamental analysis (P/E, ROE, P/B, D/E, revenue growth) — score 0-100
- Composite score = 65% technical + 35% fundamental
- Daily email report (Europe + US combined) at 8h00 Mon-Fri
- Portfolio management with ATR stop-loss alerts
- Backtesting engine (score ≥ 75 → buy, exit at ATR stop-loss or 20 days)
- 3 user levels: beginner / intermediate / expert (adapts UI and email content)
- Theme switcher: dark / light / beige (CSS custom properties, localStorage)

## Tech stack

- **Backend**: FastAPI, SQLAlchemy (SQLite), APScheduler, yfinance, `ta` library
- **ML**: scikit-learn RandomForestClassifier, joblib
- **Frontend**: Bootstrap 5, Jinja2 templates, custom CSS with CSS variables
- **Email**: Gmail SMTP, port 587
- **Auth**: JWT cookie, pure Python `hashlib.pbkdf2_hmac("sha256", ..., 600_000)` — no bcrypt/passlib
- **Python**: 3.9+ (use `from __future__ import annotations` in ALL new files)

## Project structure

```
app/
  config.py          # ENV config: DATABASE_URL, SECRET_KEY, EMAIL_*, ROLLING_WINDOW=200
  database.py        # SQLAlchemy engine, SessionLocal, init_db() + SQLite migrations
  models.py          # ORM models: User, Stock, DailyData, AnalysisResult, PortfolioPosition, Alert
  tickers.py         # CAC40, SBF120, SP500 (GitHub CSV), NASDAQ (HTTP FTP) ticker lists
  data_engine.py     # yfinance batch download (BATCH_SIZE=20), rolling 200-day window
  indicators.py      # compute_indicators(): RSI, MACD, ATR, BB, OBV, Ichimoku, SMA/EMA
  scoring.py         # compute_score(ind, ml_prob) → (score_base, ml_boost, score_final, ranking)
  ml_model.py        # RandomForest: train(), predict(), load_metrics(), save_metrics()
  fundamentals.py    # fetch yfinance .info, compute_fundamental_score(), update_fundamentals()
  backtest.py        # run_backtest() → MarketStats per market + GLOBAL aggregate
  email_sender.py    # send_combined_report(), send_stop_loss_alert() via Gmail SMTP
  auth.py            # hash_password(), verify_password(), JWT cookie auth, get_current_user()
  scheduler.py       # APScheduler cron jobs (see schedule below)
  main.py            # FastAPI app, routers, /admin/* endpoints
  routers/
    auth_router.py   # /login, /logout, /register
    dashboard.py     # /dashboard — score table with fundamental columns
    portfolio.py     # /portfolio, /portfolio/add, /portfolio/import, /portfolio/delete/{id}
    backtest_router.py  # /backtest — loads ml_models/backtest_cache.json
templates/
  base.html          # Navbar + theme switcher (dark/light/beige) + localStorage JS
  dashboard.html     # Score table, adapts columns by user.level
  portfolio.html     # Positions table + add/import modals
  backtest.html      # Global KPIs + per-market bt-card layout
  login.html         # Standalone (no base.html), data-theme="dark"
  register.html      # Standalone (no base.html), data-theme="dark"
static/
  style.css          # CSS custom properties for 3 themes, all component classes
ml_models/
  rf_model.pkl            # trained RandomForest (gitignored)
  scaler.pkl              # StandardScaler (gitignored)
  rf_model_metrics.json   # accuracy, AUC, n_samples (committed)
  backtest_cache.json     # backtest results cache (committed)
```

## Scheduled jobs (Europe/Paris timezone)

| Job | When | Duration |
|-----|------|----------|
| `job_update_and_score` | Mon-Sat 02h00 | ~30-60 min |
| `job_retrain_ml` | Sun 02h00 | ~5 min |
| `job_update_fundamentals` | Sun 03h00 | ~6 min (667 × 0.4s) |
| `job_email_daily` | Mon-Fri 08h00 | <1 min |
| `job_check_stop_losses` | Mon-Fri every 15min 09h-22h | <1 min |

## Admin endpoints (no auth)

- `GET /admin/run-now` — trigger data update + scoring in background
- `GET /admin/fundamentals-now` — trigger fundamental fetch in background
- `POST /admin/backtest-run` — trigger backtest in background → redirects to /backtest

## Scoring system

**Technical score (0-100):**
- Trend (SMA50 > SMA200, EMA50): 25 pts
- Momentum (RSI, MACD): 25 pts
- Volume/OBV: 20 pts
- Ichimoku: 15 pts
- ML boost: -15 to +15 pts

**Fundamental score (0-100):**
- P/E ratio: 30 pts (< 15 = excellent)
- ROE: 25 pts (> 15% = good)
- P/B ratio: 20 pts
- D/E ratio: 15 pts (yfinance returns × 100, e.g. 150 = 1.5×)
- Revenue growth: 10 pts

**Composite score = 65% technical + 35% fundamental**

**Rankings:** Strong Buy (≥ 75) | Buy (≥ 58) | Neutral (≥ 42) | Avoid (< 42)

## ML model

- Features: RSI, MACD_hist, BB_pct, Vol_ratio, ATR/close, ROC, OBV_norm, SMA50/close, SMA200/close, Ichimoku_bull
- Label: close +2% in 10 days → binary
- Split: 80/20 chronological (no shuffling)
- Key fix: `SMA50 = close.rolling(50, min_periods=20)`, `SMA200 = close.rolling(200, min_periods=50)` — prevents data starvation

## Database migrations

New columns are added via `_migrate_fundamental_columns()` in `database.py` on startup.
Pattern: try each `ALTER TABLE analysis_results ADD COLUMN` wrapped in try/except (SQLite doesn't support `IF NOT EXISTS`).

## CSS theming

All colors use CSS custom properties defined per theme:
```css
[data-theme="dark"]  { --bg: #0f172a; --text: #e2e8f0; ... }
[data-theme="light"] { --bg: #f1f5f9; --text: #0f172a; ... }
[data-theme="beige"] { --bg: #f5f0e8; --text: #2c2416; ... }
```
Theme persisted in `localStorage`. Set on `<html data-theme="dark">` as default.
Never use `table-dark`, `bg-dark`, or hardcoded Bootstrap dark classes — use CSS variables instead.

## Key rules

- Always add `from __future__ import annotations` at the top of every new Python file (Python 3.9 compatibility for `X | Y` union types).
- Never use `bcrypt` or `passlib` — use `hashlib.pbkdf2_hmac("sha256", ...)` directly.
- yfinance batch downloads: `BATCH_SIZE=20` max to avoid JSONDecodeError. Individual fallback for failed tickers.
- yfinance MultiIndex handling: always flatten with `df.columns = df.columns.get_level_values(0)` after batch download.
- SQLAlchemy `.scalar()` must always be preceded by `.limit(1)` to avoid `MultipleResultsFound`.
- Rolling window = 200 days. New data: insert + delete oldest row if count > 200.
- Rate limiting on yfinance `.info` calls: `time.sleep(0.4)` between tickers.

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set env variables (copy .env.example → .env, fill in values)
cp .env.example .env

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8006 --reload

# Trigger initial data load (30-60 min)
curl http://localhost:8006/admin/run-now

# Trigger fundamental analysis (after data is loaded, ~6 min)
curl http://localhost:8006/admin/fundamentals-now

# Train ML model (after data is loaded)
# Happens automatically Sunday 02h00, or via scheduler job
```

## Environment variables

```env
DATABASE_URL=sqlite:///./stocks.db
SECRET_KEY=<long-random-string>
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your@gmail.com
EMAIL_PASSWORD=<gmail-app-password>
EMAIL_FROM=your@gmail.com
```

## Known gotchas

- SP500 tickers fetched from GitHub CSV (Wikipedia blocks scraping). Fallback: hardcoded 503-ticker list.
- NASDAQ tickers fetched via HTTP FTP (SSL issues with HTTPS). Fallback: 49-ticker list.
- yfinance `debtToEquity` returns value × 100 (e.g., 150.0 = 1.5× D/E ratio). Dashboard divides by 100 for display.
- Backtest cache is a JSON file (`ml_models/backtest_cache.json`). Re-run via `/admin/backtest-run` after new data.
- Fundamentals are fetched weekly (Sunday). Show `—` in dashboard until first fetch completes.
