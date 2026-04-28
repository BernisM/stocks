from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    level         = Column(String, default="intermediate")  # beginner | intermediate | expert
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    positions = relationship("PortfolioPosition", back_populates="user", cascade="all, delete")
    alerts    = relationship("Alert", back_populates="user", cascade="all, delete")


class Stock(Base):
    __tablename__ = "stocks"

    id           = Column(Integer, primary_key=True, index=True)
    ticker       = Column(String, unique=True, index=True, nullable=False)
    name         = Column(String, default="")
    isin         = Column(String, nullable=True)
    market       = Column(String, nullable=False)   # CAC40 | SBF120 | SP500 | COMMODITIES
    sector       = Column(String, nullable=True)
    last_updated = Column(DateTime, nullable=True)

    daily_data = relationship("DailyData", back_populates="stock", cascade="all, delete")
    analyses   = relationship("AnalysisResult", back_populates="stock", cascade="all, delete")


class DailyData(Base):
    __tablename__ = "daily_data"
    __table_args__ = (UniqueConstraint("stock_id", "date"),)

    id       = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False, index=True)
    date     = Column(DateTime, nullable=False, index=True)
    open     = Column(Float)
    high     = Column(Float)
    low      = Column(Float)
    close    = Column(Float)
    volume   = Column(Float)

    stock = relationship("Stock", back_populates="daily_data")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    __table_args__ = (UniqueConstraint("stock_id", "date"),)

    id              = Column(Integer, primary_key=True, index=True)
    stock_id        = Column(Integer, ForeignKey("stocks.id"), nullable=False, index=True)
    date            = Column(DateTime, nullable=False, index=True)

    # Prix & risque
    close           = Column(Float)
    atr             = Column(Float)
    stop_loss_price = Column(Float)
    volatility      = Column(Float)   # % annualisé

    # Indicateurs
    rsi             = Column(Float)
    macd            = Column(Float)
    macd_signal     = Column(Float)
    macd_hist       = Column(Float)
    bollinger_b     = Column(Float)
    ema50           = Column(Float)
    sma200          = Column(Float)
    volume_ratio    = Column(Float)

    # Score & ML
    score_base      = Column(Integer)   # 0-85 (hors ML)
    ml_probability  = Column(Float)     # 0-1
    ml_boost        = Column(Integer)   # -15 à +15
    score_final     = Column(Integer)   # 0-100
    ranking         = Column(String)    # Strong Buy | Buy | Neutral | Avoid

    # Indicateurs avancés
    adx            = Column(Float, nullable=True)
    sma200_slope   = Column(Float, nullable=True)
    atr_pct_rank   = Column(Float, nullable=True)
    bb_zscore      = Column(Float, nullable=True)

    # Analyse fondamentale (mise à jour hebdomadaire)
    fundamental_score = Column(Integer, nullable=True)   # 0-100
    score_composite   = Column(Integer, nullable=True)   # 65% tech + 35% fonda
    pe_ratio          = Column(Float,   nullable=True)   # P/E
    pb_ratio          = Column(Float,   nullable=True)   # P/B
    roe               = Column(Float,   nullable=True)   # Return on Equity %
    debt_equity       = Column(Float,   nullable=True)   # D/E ratio (× 100)
    rev_growth        = Column(Float,   nullable=True)   # Revenue growth %
    peg_ratio         = Column(Float,   nullable=True)   # PEG ratio
    ev_ebit           = Column(Float,   nullable=True)   # EV/EBIT
    ev_ebitda         = Column(Float,   nullable=True)   # EV/EBITDA
    fcf               = Column(Float,   nullable=True)   # Free Cash Flow (valeur brute)

    # Hyper-Growth — détection licornes potentielles (None si non éligible)
    hyper_growth_score = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    stock      = relationship("Stock", back_populates="analyses")


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ticker          = Column(String, nullable=False)
    name            = Column(String, default="")
    shares          = Column(Float, nullable=False)
    buy_price       = Column(Float, nullable=False)
    buy_date        = Column(DateTime, nullable=False)
    stop_loss_price = Column(Float, nullable=True)
    fees            = Column(Float, default=0.0, nullable=True)   # frais de courtage
    notes           = Column(Text, default="")
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    user   = relationship("User", back_populates="positions")
    alerts = relationship("Alert", back_populates="position", cascade="all, delete")


class ExtraRecipient(Base):
    __tablename__ = "extra_recipients"

    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String, unique=True, index=True, nullable=False)
    name       = Column(String, default="")
    level      = Column(String, default="beginner")  # beginner | intermediate | expert
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Dividend(Base):
    __tablename__ = "dividends"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ticker     = Column(String, nullable=False)
    name       = Column(String, default="")
    amount     = Column(Float, nullable=False)
    date       = Column(DateTime, nullable=False)
    notes      = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


class UserEvent(Base):
    __tablename__ = "user_events"

    id         = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, nullable=True, index=True)
    event_type = Column(String, nullable=False, index=True)  # login_ok | login_fail | register | page | action
    detail     = Column(String, default="")
    ip         = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Alert(Base):
    __tablename__ = "alerts"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    position_id = Column(Integer, ForeignKey("portfolio_positions.id"), nullable=True)
    type        = Column(String)     # stop_loss | buy_signal | sell_signal
    message     = Column(Text)
    is_read     = Column(Boolean, default=False)
    sent_at     = Column(DateTime, default=datetime.utcnow)

    user     = relationship("User", back_populates="alerts")
    position = relationship("PortfolioPosition", back_populates="alerts")
