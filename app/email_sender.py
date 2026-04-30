"""
Envoi des emails via Gmail SMTP.
Contenu adapté au niveau de l'utilisateur : beginner | intermediate | expert.
"""
from __future__ import annotations
import logging
import smtplib
import socket
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .config import EMAIL_FROM, EMAIL_HOST, EMAIL_PASSWORD, EMAIL_PORT, EMAIL_USER
from .scoring import RANKING_EMOJI

logger = logging.getLogger(__name__)


# ── SMTP helper ───────────────────────────────────────────────────────────────

def _send(to: str | list[str], subject: str, html_body: str) -> None:
    if not EMAIL_USER:
        logger.warning("EMAIL_USER non configuré — email non envoyé")
        return

    recipients = [to] if isinstance(to, str) else to
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        # Force IPv4 pour éviter ENETUNREACH si Render n'a pas de route IPv6
        ipv4 = socket.getaddrinfo(EMAIL_HOST, EMAIL_PORT, socket.AF_INET)[0][4][0]
        with smtplib.SMTP(ipv4, EMAIL_PORT) as smtp:
            smtp.ehlo(EMAIL_HOST)
            smtp.starttls()
            smtp.login(EMAIL_USER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_FROM, recipients, msg.as_string())
        logger.info(f"Email envoyé à {recipients}")
    except Exception as e:
        logger.error(f"Erreur envoi email: {e}")


# ── Construction du tableau HTML ──────────────────────────────────────────────

def _row_color(ranking: str) -> str:
    return {
        "Strong Buy": "#1a3a1a",
        "Buy":        "#1e2e1e",
        "Neutral":    "#2a2a2a",
        "Avoid":      "#3a1a1a",
    }.get(ranking, "#2a2a2a")


def _build_table(rows: list[dict], level: str) -> str:
    headers_base  = ["#", "Ticker", "Nom", "Prix", "Score", "Signal", "Stop-Loss"]
    headers_inter = headers_base + ["RSI", "MACD ▲▼", "Volatilité"]
    headers_exp   = headers_inter + ["ML prob.", "ATR%", "Bollinger %B"]

    headers = {
        "beginner":     headers_base,
        "intermediate": headers_inter,
        "expert":       headers_exp,
    }.get(level, headers_inter)

    def th(h: str) -> str:
        return f'<th style="padding:8px 12px;text-align:left;border-bottom:1px solid #444">{h}</th>'

    def td(v: str, align: str = "left") -> str:
        return f'<td style="padding:8px 12px;text-align:{align}">{v}</td>'

    head_html = "".join(th(h) for h in headers)
    body_html = ""

    for i, r in enumerate(rows, 1):
        bg      = _row_color(r["ranking"])
        emoji   = RANKING_EMOJI.get(r["ranking"], "")
        signal  = f'{emoji} {r["ranking"]}'
        stop_pct = ((r["stop_loss"] - r["close"]) / r["close"] * 100) if r["close"] else 0

        cells = [
            td(str(i), "center"),
            td(f'<strong>{r["ticker"]}</strong>'),
            td((r.get("name", "") or "")[:22], "left"),
            td(f'{r["close"]:.2f}', "right"),
            td(f'<strong>{r["score_final"]}/100</strong>', "center"),
            td(signal),
            td(f'{r["stop_loss"]:.2f} <span style="color:#f87171">({stop_pct:.1f}%)</span>', "right"),
        ]

        if level in ("intermediate", "expert"):
            macd_arrow = "▲" if r.get("macd_hist", 0) > 0 else "▼"
            macd_color = "#4ade80" if r.get("macd_hist", 0) > 0 else "#f87171"
            cells += [
                td(f'{r.get("rsi", 0):.1f}', "center"),
                td(f'<span style="color:{macd_color}">{macd_arrow}</span>', "center"),
                td(f'{r.get("volatility", 0):.1f}%', "right"),
            ]

        if level == "expert":
            ml_prob = r.get("ml_probability")
            ml_txt  = f'{ml_prob*100:.1f}%' if ml_prob is not None else "N/A"
            cells += [
                td(ml_txt, "center"),
                td(f'{r.get("atr_pct", 0):.2f}%', "right"),
                td(f'{r.get("bollinger_b", 0):.2f}', "right"),
            ]

        row_html = "".join(cells)
        body_html += f'<tr style="background:{bg}">{row_html}</tr>'

    return f"""
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch">
    <table style="width:100%;min-width:600px;border-collapse:collapse;font-size:13px">
      <thead><tr style="background:#1e293b">{head_html}</tr></thead>
      <tbody>{body_html}</tbody>
    </table>
    </div>"""


def _beginner_legend() -> str:
    return """
    <div style="margin-top:16px;padding:12px;background:#1e293b;border-radius:8px;font-size:13px">
      <strong>Comment lire ce tableau ?</strong><br>
      🔥 <strong>Strong Buy</strong> : signal très fort, tous les indicateurs sont alignés.<br>
      🟢 <strong>Buy</strong> : bon signal, tendance haussière confirmée.<br>
      ⚪ <strong>Neutral</strong> : pas de signal clair — attendre.<br>
      🔴 <strong>Avoid</strong> : signaux négatifs — ne pas acheter.<br>
      <em>Stop-Loss</em> : niveau de prix auquel vendre pour limiter les pertes (calculé avec l'ATR).
    </div>"""


# ── Emails de rapport ─────────────────────────────────────────────────────────

def send_daily_report(
    recipients: list[tuple[str, str]],   # [(email, level), ...]
    top_rows: list[dict],
    market: str,
    session_label: str,                  # "🌍 Europe" ou "🇺🇸 US"
    ml_metrics: dict | None = None,
) -> None:
    date_str = datetime.now().strftime("%d %B %Y")
    subject  = f"📊 Analyse {session_label} — Top 10 {market} — {date_str}"

    for email, level in recipients:
        table = _build_table(top_rows, level)
        legend = _beginner_legend() if level == "beginner" else ""

        ml_html = ""
        if ml_metrics and level == "expert":
            ml_html = f"""
            <div style="margin-top:12px;font-size:12px;color:#94a3b8">
              Modèle ML — Accuracy: {ml_metrics.get('accuracy','N/A')}% |
              AUC: {ml_metrics.get('auc','N/A')}% |
              Entraîné sur {ml_metrics.get('n_samples','?'):,} observations
            </div>"""

        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px">
          <h2 style="color:#38bdf8">📊 Analyse {session_label} — Top 10 {market}</h2>
          <p style="color:#94a3b8">{date_str}</p>
          {table}
          {legend}
          {ml_html}
          <hr style="border-color:#334155;margin-top:24px">
          <p style="font-size:11px;color:#475569">
            Ce rapport est fourni à titre informatif uniquement.
            Il ne constitue pas un conseil en investissement.
          </p>
        </body></html>"""

        _send(email, subject, html)


# ── Email combiné Europe + US ─────────────────────────────────────────────────

def send_combined_report(
    recipients: list[tuple[str, str]],
    top_cac40: list[dict],
    top_sbf120: list[dict],
    top_sp500: list[dict],
    analysis_date,
    ml_metrics: dict | None = None,
    top_commodities: list[dict] | None = None,
    top_crypto: list[dict] | None = None,
    session: str = "morning",   # "morning" | "afternoon"
    market_status: dict | None = None,
) -> None:
    date_str = analysis_date.strftime("%A %d %B %Y") if hasattr(analysis_date, "strftime") else str(analysis_date)
    if session == "afternoon":
        subject      = f"📊 Séance US du {date_str} — S&P500, Matières Premières & Crypto"
        session_note = """
        <div style="background:#1e293b;border-left:3px solid #f59e0b;padding:10px 14px;border-radius:4px;margin-bottom:20px;font-size:13px;color:#94a3b8">
          🕒 <strong>Prix mis à jour à 15h35</strong> — S&P500 en début de séance US ·
          CAC40/SBF120 en cours de clôture Europe
        </div>"""
    else:
        subject      = f"📊 Analyse du {date_str} — CAC40, SBF120, S&P500, Matières Premières & Crypto"
        session_note = """
        <div style="background:#1e293b;border-left:3px solid #38bdf8;padding:10px 14px;border-radius:4px;margin-bottom:20px;font-size:13px;color:#94a3b8">
          🌅 <strong>Prix d'ouverture Europe</strong> pour CAC40/SBF120 ·
          <strong>Dernière clôture US</strong> pour S&P500
        </div>"""

    ms = market_status or {}

    def _qs(market_key: str) -> str:
        d = ms.get(market_key, {})
        if not d.get("display"):
            return ""
        dot = "🟢" if d.get("market_state") == "REGULAR" else "⚫"
        return f' <span style="font-size:12px;color:#64748b;font-weight:normal">{dot} {d["display"]}</span>'

    for email, level in recipients:
        def _table(rows, label, color, qs=""):
            if not rows:
                return f"<p style='color:#94a3b8'>Aucun signal {label} aujourd'hui.</p>"
            return (
                f'<h3 style="color:#e2e8f0;border-left:3px solid {color};padding-left:10px;margin-top:28px">'
                f'{label}{qs}</h3>' + _build_table(rows, level)
            )

        body = (
            _table(top_cac40,  "🇫🇷 Top CAC40",           "#38bdf8", _qs("CAC40")) +
            _table(top_sbf120, "🌍 Top SBF120",           "#34d399", _qs("SBF120")) +
            _table(top_sp500,  "🇺🇸 Top S&P500",          "#f59e0b", _qs("SP500")) +
            (_table(top_commodities, "🪙 Matières Premières", "#fb923c", _qs("COMMODITIES")) if top_commodities else "") +
            (_table(top_crypto, "₿ Cryptomonnaies",      "#a78bfa", _qs("CRYPTO")) if top_crypto else "")
        )
        legend = _beginner_legend() if level == "beginner" else ""

        ml_html = ""
        if ml_metrics and level == "expert":
            ml_html = f"""
            <div style="margin-top:8px;font-size:12px;color:#64748b">
              🤖 ML — Accuracy {ml_metrics.get('accuracy','N/A')}% |
              AUC {ml_metrics.get('auc','N/A')}% |
              {ml_metrics.get('n_samples','?'):,} observations
            </div>"""

        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;max-width:900px;margin:auto">
          <h2 style="color:#38bdf8;margin-bottom:4px">📊 Analyse quotidienne</h2>
          <p style="color:#64748b;margin-top:0">{date_str}</p>
          {session_note}
          {body}
          {legend}
          {ml_html}
          <div style="margin-top:28px;text-align:center">
            <a href="https://stocks-ninq.onrender.com/dashboard?market=SBF120&ranking="
               style="display:inline-block;padding:10px 24px;background:#38bdf8;color:#0f172a;text-decoration:none;border-radius:6px;font-weight:bold">
              📊 Voir le dashboard complet
            </a>
          </div>
          <hr style="border-color:#1e293b;margin-top:32px">
          <p style="font-size:11px;color:#334155">
            Rapport généré automatiquement — à titre informatif uniquement, ne constitue pas un conseil en investissement.
          </p>
        </body></html>"""

        _send(email, subject, html)


# ── Alert stop-loss ───────────────────────────────────────────────────────────

def send_stop_loss_alert(
    email: str,
    ticker: str,
    current_price: float,
    stop_price: float,
    buy_price: float,
    level: str = "intermediate",
) -> None:
    pnl_pct = (current_price - buy_price) / buy_price * 100
    pnl_color = "#4ade80" if pnl_pct >= 0 else "#f87171"

    detail = ""
    if level != "beginner":
        detail = f"""
        <p>Prix d'achat : <strong>{buy_price:.2f}</strong><br>
        Stop-loss ATR : <strong>{stop_price:.2f}</strong><br>
        P&L : <span style="color:{pnl_color}"><strong>{pnl_pct:+.2f}%</strong></span></p>"""

    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px">
      <h2 style="color:#f87171">🚨 Alerte Stop-Loss : {ticker}</h2>
      <p>Le prix actuel (<strong>{current_price:.2f}</strong>) a franchi votre stop-loss.</p>
      {detail}
      <p style="background:#1e293b;padding:12px;border-radius:8px">
        ⚠️ Envisagez de vendre votre position pour limiter vos pertes.
      </p>
    </body></html>"""

    _send(email, f"🚨 Stop-Loss déclenché : {ticker}", html)


# ── Email diff de tickers (refresh hebdo) ─────────────────────────────────────

def send_ticker_diff_alert(to: str | list[str], diffs: dict[str, dict]) -> None:
    """
    Envoie un email récapitulatif des changements de tickers détectés
    par le refresh hebdomadaire (CAC40, SBF120, NASDAQ_GROWTH).
    Skip l'envoi si aucun changement.
    """
    has_changes = any(
        d.get("source_ok") and (d.get("added") or d.get("removed"))
        for d in diffs.values()
    )
    if not has_changes:
        logger.info("[ticker_diff] aucun changement — email non envoyé")
        return

    sections = []
    total_added = total_removed = 0
    for market, diff in diffs.items():
        if not diff.get("source_ok"):
            sections.append(
                f'<div style="margin-bottom:18px"><strong>{market}</strong> : '
                f'<span style="color:#f87171">⚠️ source indisponible</span></div>'
            )
            continue
        added   = diff.get("added", [])
        removed = diff.get("removed", [])
        if not added and not removed:
            continue
        total_added   += len(added)
        total_removed += len(removed)

        added_html   = ", ".join(f'<code>{t}</code>' for t in added)   or "<em>—</em>"
        removed_html = ", ".join(f'<code>{t}</code>' for t in removed) or "<em>—</em>"

        sections.append(f"""
        <div style="margin-bottom:20px;padding:14px;background:#1e293b;border-radius:8px">
          <div style="font-weight:bold;font-size:15px;margin-bottom:8px">📊 {market}
            <span style="color:#94a3b8;font-size:12px;font-weight:normal">— total {diff['total']} tickers</span>
          </div>
          <div style="margin:6px 0"><span style="color:#4ade80">➕ Ajouts ({len(added)})</span> : {added_html}</div>
          <div style="margin:6px 0"><span style="color:#f87171">➖ Retraits ({len(removed)})</span> : {removed_html}</div>
        </div>""")

    if not sections:
        return

    date_str = datetime.now().strftime("%d/%m/%Y")
    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px">
      <h2>📋 Mise à jour des tickers — {date_str}</h2>
      <p style="color:#94a3b8;font-size:13px">
        Détecté automatiquement par le refresh hebdomadaire (Wikipedia + iShares IWO).
        Total : <strong style="color:#4ade80">+{total_added}</strong> ajouts,
        <strong style="color:#f87171">-{total_removed}</strong> retraits.
      </p>
      {''.join(sections)}
      <p style="font-size:12px;color:#94a3b8;margin-top:24px">
        Les tickers ajoutés seront synchronisés au prochain run-now.
        Les tickers retirés sont marqués inactifs (soft-delete) — leur historique reste conservé.
      </p>
    </body></html>"""

    subject = f"📋 Tickers : +{total_added} ajouts, -{total_removed} retraits"
    _send(to, subject, html)
