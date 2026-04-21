# StockAnalyzer — CLAUDE.md

## Project overview

Full-stack stock analysis platform built with FastAPI + SQLite (PostgreSQL on Render).
Analyzes ~667 stocks across CAC40, SBF120, SP500 (NASDAQ removed) with:
- Technical indicators (RSI, MACD, ATR, Bollinger Bands, OBV, Ichimoku)
- RandomForest ML scoring (~66.8% accuracy, 100k+ observations)
- Fundamental analysis (P/E, ROE, P/B, D/E, revenue growth) — score 0-100
- Composite score = 65% technical + 35% fundamental
- Daily email report (CAC40 + SBF120 + S&P500 combined) at 09h15 via cron-job.org
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
- **Scheduling**: cron-job.org (external) for data/email jobs; APScheduler (internal) for stop-loss checks only
- **Markets**: CAC40, SBF120, SP500, COMMODITIES (13 futures: GC=F, SI=F, CL=F, BZ=F, NG=F, HG=F, PL=F, PA=F, ZW=F, ZC=F, KC=F, SB=F, CC=F)
- **Python**: 3.9+ (use `from __future__ import annotations` in ALL new files)

## Project structure

```
app/
  config.py          # ENV config: DATABASE_URL, SECRET_KEY, EMAIL_*, ROLLING_WINDOW=200
  database.py        # SQLAlchemy engine, SessionLocal, init_db() + migrations:
                     #   _migrate_fundamental_columns(), _migrate_portfolio_columns()
  models.py          # ORM models: User, Stock (+ isin), DailyData, AnalysisResult,
                     #   PortfolioPosition (+ fees), Dividend, ExtraRecipient, Alert
  tickers.py         # CAC40, SBF120, SP500 (GitHub CSV), COMMODITIES (13 futures) — NASDAQ removed
  data_engine.py     # yfinance batch download (BATCH_SIZE=20), rolling 200-day window
  indicators.py      # compute_indicators(): RSI, MACD, ATR, BB, OBV, Ichimoku, SMA/EMA
  scoring.py         # compute_score(ind, ml_prob) → (score_base, ml_boost, score_final, ranking)
  ml_model.py        # RandomForest: train(), predict(), load_metrics(), save_metrics()
  fundamentals.py    # fetch yfinance .info, compute_fundamental_score(), update_fundamentals()
                     # Skips COMMODITIES market. Populates stock.name from longName/shortName.
  backtest.py        # run_backtest() → MarketStats per market + GLOBAL aggregate
                     # Entry: SCORE_BUY=80, MIN_FUNDAMENTAL=40 filter
  email_sender.py    # send_combined_report(..., top_commodities=[]) — optional 4th market section
                     # send_stop_loss_alert() via Gmail SMTP
  auth.py            # hash_password(), verify_password(), JWT cookie auth, get_current_user()
                     # All auth failures → RedirectResponse("/login") not JSON 401
  scheduler.py       # APScheduler: only job_check_stop_losses (Mon-Fri every 15min 9h-22h)
                     # Email/data jobs removed — handled by cron-job.org
  main.py            # FastAPI app, routers, /admin/* endpoints, sync/job-status endpoints
  routers/
    auth_router.py      # /login, /logout, /register
    dashboard.py        # /dashboard — score table with fundamental columns + column filters
    portfolio.py        # /portfolio, /portfolio/add, /portfolio/import, /portfolio/delete/{id}
                        # /portfolio/dividends/add, /portfolio/dividends/delete/{id}
    backtest_router.py  # /backtest — loads ml_models/backtest_cache.json
    recipients_router.py # /recipients (owner-only), /recipients/add, /recipients/delete/{id}
    guide_router.py     # /guide — explains all indicators, adapts to user.level
    stocks_router.py    # /stocks/search?q= (ticker/name/isin search → JSON)
                        # /stocks/template-csv?tickers=A,B,C (pre-filled portfolio CSV download)
templates/
  base.html          # Navbar + theme switcher (dark/light/beige) + localStorage JS
  dashboard.html     # Score table with per-column text/numeric filter row + smart Rapport modal
  portfolio.html     # Positions table (+ fees col) + dividends section + add/import modals
  backtest.html      # Global KPIs + per-market bt-card layout
  guide.html         # Indicator explanations + 🔍 stock search & CSV export section
  recipients.html    # Extra email recipients management (owner-only)
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

## Scheduled jobs

### cron-job.org (external — triggers Render server)

| Job | URL | Time |
|-----|-----|------|
| Wake up (keep-alive) | `/ping` | Every ~14 min (prevents Render sleep) |
| Sync Daily Prices | `/admin/run-now` | 09h05 daily |
| Train ML | `/admin/train-ml` | 09h12 daily |
| Fondamentaux | `/admin/fundamentals-now` | 09h14 daily |
| Email rapport | `/admin/send-email` | 09h15 daily |

### APScheduler (internal — only stop-loss)

| Job | When | Duration |
|-----|------|----------|
| `job_check_stop_losses` | Mon-Fri every 15min 09h-22h | <1 min |

**Note:** APScheduler email job was removed to avoid double-sending (cron-job.org handles it).

## Admin endpoints (no auth required)

- `GET /ping` — public keep-alive, returns `{"status": "ok"}`
- `GET /admin/run-now` — trigger data update + scoring in background (~30-60 min)
- `GET /admin/train-ml` — trigger ML retraining in background (~5 min)
- `GET /admin/fundamentals-now` — trigger fundamental fetch in background (~6 min)
- `GET /admin/send-email` — trigger daily email report
- `POST /admin/backtest-run` — trigger backtest → redirects to /backtest

## Authenticated endpoints (require JWT cookie)

- `POST /sync-prices` — manual sync trigger with progress tracking
- `GET /sync-status` — returns `{running, phase, progress, total, pct, error}`
- `GET /job-status` — returns `{sync_prices, train_ml, fondamentaux}` ISO timestamps of last completion
- `GET /smart-email-status` — returns `{running, jobs_done, email_sent, error}`
- `POST /send-email-smart` — runs selected jobs then sends email; body: `{jobs: ["sync_prices", "train_ml", "fondamentaux"]}`
- `GET /send-email-me` — sends report to logged-in user + all ExtraRecipients
- `GET /stocks/search?q=` — search stocks by ticker/name/isin, returns JSON list
- `GET /stocks/template-csv?tickers=A,B,C` — download pre-filled portfolio CSV for selected tickers
- `POST /portfolio/dividends/add` — add a dividend record (ticker, name, amount, date, notes)
- `POST /portfolio/dividends/delete/{id}` — delete a dividend record

## In-memory state (main.py)

```python
_JOB_TIMES = {"sync_prices": None, "train_ml": None, "fondamentaux": None}
_SMART_STATE = {"running": False, "jobs_done": [], "email_sent": False, "error": None}
_sync_state = {"running": False, "phase": "", "progress": 0, "total": 0, ...}
```

## Dashboard features

- Market filter (CAC40 / SBF120 / SP500) + Signal filter (form GET)
- Per-column live filter row (JS, no reload): comma = OR logic — `Buy,Strong Buy` / `>50,<30` / `>=75` / `=80,=85`
- Smart Rapport button: checks job freshness (1h threshold), shows modal if stale, polls `/smart-email-status`
- Admin buttons: 🔄 Sync Prix, 🤖 Train ML, 📊 Fondamentaux (inline async, no reload)
- Columns adapt to user.level: beginner (basic) / intermediate (+RSI, MACD, P/E, ROE, D/E) / expert (+ML prob, P/B, Croiss%, ATR%, %B)

## Email report

`send_combined_report` signature:
```python
def send_combined_report(
    recipients: list[tuple[str, str]],  # (email, level)
    top_cac40: list[dict],
    top_sbf120: list[dict],
    top_sp500: list[dict],
    analysis_date,
    ml_metrics: dict | None = None,
) -> None
```
- 3 separate market sections (CAC40, SBF120, S&P500), top 10 each
- Mobile-safe: `overflow-x: auto` wrapper + `min-width: 600px` table
- Names truncated to 22 chars
- Dashboard link button at bottom

## Portfolio models

**PortfolioPosition** — columns: id, user_id, ticker, name, shares, buy_price, buy_date, `fees` (Float, courtage déduit du P&L net), stop_loss_price, notes, is_active

**Dividend** — columns: id, user_id, ticker, name, `amount` (Float), date, notes, created_at
- Managed via portfolio page (💰 section + add modal)
- `total_dividends` shown in summary card

**Stock** — columns: id, ticker, name, `isin` (nullable), market, last_updated
- isin added for guide search; not auto-populated (set manually or future enhancement)

## ExtraRecipient model

```python
class ExtraRecipient(Base):
    __tablename__ = "extra_recipients"
    id, email (unique), name, level, created_at
```
Managed via `/recipients` page (owner-only — `user.email == EMAIL_USER`).
Recipients = active Users + ExtraRecipients in all email sends.

## Scoring system

**Technical score (0-100):**
- Trend (SMA50 > SMA200, EMA50): 25 pts
- Momentum RSI: +12 if RSI 50–70 | +8 if RSI <30 AND MACD_hist>0 (reversal confirmed) | 0 if RSI <30 alone (falling knife) | −5 if RSI >70
- Momentum MACD: +13 if MACD_hist > 0
- Volume: +10 if vol_ratio ≥ 1.5 | +5 if vol_ratio ≥ 1.3 (raised from 1.0)
- OBV: +10 if OBV slope > 0
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

**Backtest entry rules:** `SCORE_BUY = 80` (raised from 75) | `MIN_FUNDAMENTAL = 40` (stocks below excluded)

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
- TemplateResponse API: `TemplateResponse(request, "name.html", context)` — do NOT pass `"request"` inside context dict.
- datetime: use `datetime.now(UTC).replace(tzinfo=None)` — never `datetime.utcnow()` (deprecated).
- pandas chained assignment: always `df = df.copy()` before modifying computed DataFrames.
- Auth failures: always `RedirectResponse("/login", status_code=302)` — never raise HTTP 401.

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in values
uvicorn app.main:app --host 0.0.0.0 --port 8006 --reload
curl http://localhost:8006/admin/run-now        # initial data load (~30-60 min)
curl http://localhost:8006/admin/fundamentals-now  # after data loaded (~6 min)
```

## Environment variables

```env
DATABASE_URL=sqlite:///./stocks.db
SECRET_KEY=<long-random-string>
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your@gmail.com          # also used as owner check for /recipients
EMAIL_PASSWORD=<gmail-app-password>
EMAIL_FROM=your@gmail.com
```

## Known gotchas

- SP500 tickers fetched from GitHub CSV (Wikipedia blocks scraping). Fallback: hardcoded 503-ticker list.
- NASDAQ removed entirely (was causing noise; tickers removed from `tickers.py` and dashboard markets list).
- COMMODITIES uses continuous futures tickers (`GC=F` etc.). No fundamental data fetched for them (skipped in `update_fundamentals`). Names hardcoded in `COMMODITY_NAMES` dict in `tickers.py`.
- `send_combined_report` has optional `top_commodities` param (default None) — section only appears in email if non-empty.
- yfinance `debtToEquity` returns value × 100 (e.g., 150.0 = 1.5× D/E ratio). Dashboard divides by 100 for display.
- Backtest cache is a JSON file (`ml_models/backtest_cache.json`). Re-run via `/admin/backtest-run` after new data.
- Portfolio fees: deducted from gross P&L → `pnl_abs = (current - buy) * shares - fees`. pnl_pct uses `fees / (buy * shares)` as base.
- Guide stock search uses `/stocks/search` (debounce 280ms). Multi-select → `/stocks/template-csv` downloads pre-filled CSV with ticker/name/isin columns.
- Fundamentals fetched daily at 09h14. Show `—` in dashboard until first fetch completes.
- SBF120 shows ~46 stocks instead of ~92: CAC40 stocks are stored with `market="CAC40"` and are not duplicated under SBF120. This is expected — the dashboard filter for SBF120 only shows stocks whose primary market is SBF120.
- Double email bug (fixed): APScheduler email job was removed; only cron-job.org triggers `/admin/send-email`.
- Render free tier sleeps after inactivity → cron-job.org Wake up job hits `/ping` every ~14 min.
