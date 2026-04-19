import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stocks.db")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-use-long-random-string")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 jours

EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")

ML_MODEL_PATH = "./ml_models/rf_model.pkl"
ML_SCALER_PATH = "./ml_models/scaler.pkl"

ROLLING_WINDOW = 200       # jours conservés par action
ATR_STOP_MULTIPLIER = 2.5  # stop loss = Close - 2.5 × ATR
TOP_N_EMAIL = 10           # top actions par email

# Fuseaux horaires
TZ_EUROPE = "Europe/Paris"
TZ_UTC = "UTC"
