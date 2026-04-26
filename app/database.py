from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import DATABASE_URL

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from . import models  # noqa: F401 – registers all models
    Base.metadata.create_all(bind=engine)
    _migrate_fundamental_columns()
    _migrate_portfolio_columns()
    _migrate_indicator_columns()
    _migrate_recipients_columns()
    _migrate_sector_column()
    _migrate_advanced_fundamental_columns()


def _migrate_fundamental_columns():
    """Ajoute les colonnes fondamentales si elles n'existent pas (SQLite + PostgreSQL)."""
    from sqlalchemy import text
    is_sqlite = "sqlite" in DATABASE_URL

    # SQLite uses REAL, PostgreSQL uses DOUBLE PRECISION
    float_type = "REAL" if is_sqlite else "DOUBLE PRECISION"
    new_cols = [
        ("fundamental_score", "INTEGER"),
        ("score_composite",   "INTEGER"),
        ("pe_ratio",          float_type),
        ("pb_ratio",          float_type),
        ("roe",               float_type),
        ("debt_equity",       float_type),
        ("rev_growth",        float_type),
    ]

    with engine.connect() as conn:
        if is_sqlite:
            existing = {row[1] for row in conn.execute(
                text("PRAGMA table_info(analysis_results)")
            )}
        else:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'analysis_results'"
            ))
            existing = {row[0] for row in result}

        for col_name, col_type in new_cols:
            if col_name not in existing:
                try:
                    conn.execute(text(
                        f"ALTER TABLE analysis_results ADD COLUMN {col_name} {col_type}"
                    ))
                except Exception:
                    pass
        conn.commit()


def _migrate_portfolio_columns():
    """Ajoute les colonnes fees (positions) et isin (stocks) si absentes."""
    from sqlalchemy import text
    is_sqlite = "sqlite" in DATABASE_URL
    float_type = "REAL" if is_sqlite else "DOUBLE PRECISION"

    def _existing_cols(table: str) -> set:
        with engine.connect() as conn:
            if is_sqlite:
                return {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
            ), {"t": table})
            return {row[0] for row in result}

    def _safe_alter(sql: str) -> None:
        """Exécute un ALTER TABLE dans sa propre transaction (PostgreSQL-safe)."""
        with engine.connect() as conn:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                conn.rollback()

    if "fees" not in _existing_cols("portfolio_positions"):
        _safe_alter(
            f"ALTER TABLE portfolio_positions ADD COLUMN fees {float_type} DEFAULT 0"
        )

    if "isin" not in _existing_cols("stocks"):
        _safe_alter("ALTER TABLE stocks ADD COLUMN isin VARCHAR")


def _migrate_indicator_columns():
    """Ajoute les colonnes d'indicateurs avancés si absentes."""
    from sqlalchemy import text
    is_sqlite  = "sqlite" in DATABASE_URL
    float_type = "REAL" if is_sqlite else "DOUBLE PRECISION"

    new_cols = [
        ("adx",          float_type),
        ("sma200_slope", float_type),
        ("atr_pct_rank", float_type),
        ("bb_zscore",    float_type),
    ]

    with engine.connect() as conn:
        if is_sqlite:
            existing = {row[1] for row in conn.execute(
                text("PRAGMA table_info(analysis_results)")
            )}
        else:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'analysis_results'"
            ))
            existing = {row[0] for row in result}

        for col_name, col_type in new_cols:
            if col_name not in existing:
                try:
                    conn.execute(text(
                        f"ALTER TABLE analysis_results ADD COLUMN {col_name} {col_type}"
                    ))
                except Exception:
                    pass
        conn.commit()


def _migrate_recipients_columns():
    """Ajoute is_active aux extra_recipients si absent."""
    from sqlalchemy import text
    is_sqlite = "sqlite" in DATABASE_URL
    bool_type = "INTEGER" if is_sqlite else "BOOLEAN"
    default   = "1" if is_sqlite else "TRUE"

    with engine.connect() as conn:
        if is_sqlite:
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(extra_recipients)"))}
        else:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'extra_recipients'"
            ))
            existing = {row[0] for row in result}

        if "is_active" not in existing:
            try:
                conn.execute(text(
                    f"ALTER TABLE extra_recipients ADD COLUMN is_active {bool_type} DEFAULT {default}"
                ))
                conn.commit()
            except Exception:
                conn.rollback()


def _migrate_sector_column():
    """Ajoute la colonne sector à la table stocks si absente."""
    from sqlalchemy import text
    is_sqlite = "sqlite" in DATABASE_URL

    def _existing_cols(table: str) -> set:
        with engine.connect() as conn:
            if is_sqlite:
                return {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
            ), {"t": table})
            return {row[0] for row in result}

    def _safe_alter(sql: str) -> None:
        with engine.connect() as conn:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                conn.rollback()

    if "sector" not in _existing_cols("stocks"):
        _safe_alter("ALTER TABLE stocks ADD COLUMN sector VARCHAR")


def _migrate_advanced_fundamental_columns():
    """Ajoute PEG, EV/EBIT, EV/EBITDA, FCF à analysis_results si absents."""
    from sqlalchemy import text
    is_sqlite  = "sqlite" in DATABASE_URL
    float_type = "REAL" if is_sqlite else "DOUBLE PRECISION"

    new_cols = [
        ("peg_ratio", float_type),
        ("ev_ebit",   float_type),
        ("ev_ebitda", float_type),
        ("fcf",       float_type),
    ]

    with engine.connect() as conn:
        if is_sqlite:
            existing = {row[1] for row in conn.execute(
                text("PRAGMA table_info(analysis_results)")
            )}
        else:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'analysis_results'"
            ))
            existing = {row[0] for row in result}

        for col_name, col_type in new_cols:
            if col_name not in existing:
                try:
                    conn.execute(text(
                        f"ALTER TABLE analysis_results ADD COLUMN {col_name} {col_type}"
                    ))
                except Exception:
                    pass
        conn.commit()
