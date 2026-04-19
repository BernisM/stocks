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
    market       = Column(String, nullable=False)   # CAC40 | SBF120 | SP500 | NASDAQ
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

    # Analyse fondamentale (mise à jour hebdomadaire)
    fundamental_score = Column(Integer, nullable=True)   # 0-100
    score_composite   = Column(Integer, nullable=True)   # 65% tech + 35% fonda
    pe_ratio          = Column(Float,   nullable=True)   # P/E
    pb_ratio          = Column(Float,   nullable=True)   # P/B
    roe               = Column(Float,   nullable=True)   # Return on Equity %
    debt_equity       = Column(Float,   nullable=True)   # D/E ratio (× 100)
    rev_growth        = Column(Float,   nullable=True)   # Revenue growth %

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
    notes           = Column(Text, default="")
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    user   = relationship("User", back_populates="positions")
    alerts = relationship("Alert", back_populates="position", cascade="all, delete")


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
