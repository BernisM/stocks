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

    def _existing_cols(conn, table):
        if is_sqlite:
            return {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
        ), {"t": table})
        return {row[0] for row in result}

    with engine.connect() as conn:
        pos_cols = _existing_cols(conn, "portfolio_positions")
        if "fees" not in pos_cols:
            try:
                conn.execute(text(
                    f"ALTER TABLE portfolio_positions ADD COLUMN fees {float_type} DEFAULT 0"
                ))
            except Exception:
                pass

        stk_cols = _existing_cols(conn, "stocks")
        if "isin" not in stk_cols:
            try:
                conn.execute(text("ALTER TABLE stocks ADD COLUMN isin VARCHAR"))
            except Exception:
                pass

        conn.commit()
