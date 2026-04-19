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


def _migrate_fundamental_columns():
    """Ajoute les colonnes fondamentales si elles n'existent pas encore (SQLite)."""
    new_cols = [
        ("fundamental_score", "INTEGER"),
        ("score_composite",   "INTEGER"),
        ("pe_ratio",          "REAL"),
        ("pb_ratio",          "REAL"),
        ("roe",               "REAL"),
        ("debt_equity",       "REAL"),
        ("rev_growth",        "REAL"),
    ]
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(
            __import__("sqlalchemy").text("PRAGMA table_info(analysis_results)")
        )}
        for col_name, col_type in new_cols:
            if col_name not in existing:
                conn.execute(__import__("sqlalchemy").text(
                    f"ALTER TABLE analysis_results ADD COLUMN {col_name} {col_type}"
                ))
        conn.commit()
