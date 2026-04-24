from __future__ import annotations
import threading


def log_event(
    user_email: str | None,
    event_type: str,
    detail: str = "",
    ip: str = "",
) -> None:
    """Log an event in a background thread (non-blocking)."""
    def _write():
        from .database import SessionLocal
        from .models import UserEvent
        db = SessionLocal()
        try:
            db.add(UserEvent(user_email=user_email, event_type=event_type, detail=detail, ip=ip))
            db.commit()
        except Exception:
            pass
        finally:
            db.close()
    threading.Thread(target=_write, daemon=True).start()


def log_event_sync(
    user_email: str | None,
    event_type: str,
    detail: str = "",
    ip: str = "",
) -> None:
    """Synchronous version — use in auth routes where reliability matters."""
    from .database import SessionLocal
    from .models import UserEvent
    db = SessionLocal()
    try:
        db.add(UserEvent(user_email=user_email, event_type=event_type, detail=detail, ip=ip))
        db.commit()
    except Exception:
        pass
    finally:
        db.close()
