from __future__ import annotations
"""
Fetches and caches all ticker lists: CAC40, SBF120, S&P500, NASDAQ.
"""
import io
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ── CAC 40 (composition avril 2025) ──────────────────────────────────────────
CAC40 = [
    "AI.PA",   "AIR.PA",  "ALO.PA",  "ACA.PA",  "CS.PA",   "BNP.PA",
    "EN.PA",   "CAP.PA",  "CA.PA",   "SGO.PA",  "ML.PA",   "DSY.PA",
    "EDEN.PA", "ENGI.PA", "EL.PA",   "ERF.PA",  "RMS.PA",  "KER.PA",
    "LR.PA",   "OR.PA",   "MC.PA",   "ORA.PA",  "RI.PA",   "PUB.PA",
    "RNO.PA",  "SAF.PA",  "SAN.PA",  "SU.PA",   "GLE.PA",  "STM.PA",
    "STLAM.PA","HO.PA",   "TTE.PA",  "URW.PA",  "VIE.PA",  "DG.PA",
    "VIV.PA",  "BN.PA",   "TEP.PA",  "WLN.PA",
]

# ── SBF 120 (CAC40 + 80 valeurs françaises vérifiées) ─────────────────────────
_SBF120_EXTRA = [
    # Finance & Immobilier
    "AC.PA",    # Accor
    "AMUN.PA",  # Amundi
    "COFA.PA",  # Coface
    "COV.PA",   # Covivio
    "GFC.PA",   # Gecina
    "ICAD.PA",  # Icade
    "KLE.PA",   # Klépierre
    "MF.PA",    # Wendel
    "RF.PA",    # Eurazeo
    "SCOR.PA",  # SCOR
    # Industrie & Ingénierie
    "ADP.PA",   # Aéroports de Paris
    "AKE.PA",   # Arkema
    "DEC.PA",   # JCDecaux
    "ELIS.PA",  # Elis
    "ERA.PA",   # Eramet
    "FGR.PA",   # Eiffage
    "NEX.PA",   # Nexans
    "OPM.PA",   # Opmobility (ex-Plastic Omnium)
    "RXL.PA",   # Rexel
    "SK.PA",    # SEB
    "SPIE.PA",  # SPIE
    "TKO.PA",   # Tarkett
    # Tech & Services
    "ATE.PA",   # Alten
    "BVI.PA",   # Bureau Veritas
    "FNAC.PA",  # Fnac Darty
    "GET.PA",   # Getlink
    "GTT.PA",   # GTT
    "ILD.PA",   # Iliad (Free)
    "MMB.PA",   # Lagardère
    "SOP.PA",   # Sopra Steria
    "SW.PA",    # Sodexo
    "TE.PA",    # Technip Energies
    "TFI.PA",   # TF1
    "TRI.PA",   # Trigano
    "SII.PA",   # SII Group
    # Pharma & Santé
    "BIOA.PA",  # bioMérieux
    "IPN.PA",   # Ipsen
    "VIRB.PA",  # Virbac
    "DBV.PA",   # DBV Technologies
    # Consommation & Luxe
    "BB.PA",    # BIC
    "RCO.PA",   # Rémy Cointreau
    "UBI.PA",   # Ubisoft
    "MERY.PA",  # M6 Métropole Télévision
    "ITP.PA",   # Interparfums
    # Énergie & Matériaux
    "RUI.PA",   # Rubis
    "VK.PA",    # Vicat
    "VRLA.PA",  # Verallia
    "DIM.PA",   # Sartorius Stedim Biotech
    "SOI.PA",   # Soitec
    # Divers
    "NXI.PA",   # Nexity
    "HCO.PA",   # Hexaom
]
SBF120 = list(dict.fromkeys(CAC40 + _SBF120_EXTRA))


def get_sp500() -> list[str]:
    # Essai 1 : Wikipedia avec User-Agent
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=20,
        )
        tables = pd.read_html(io.StringIO(resp.text))
        tickers = tables[0]["Symbol"].tolist()
        result = [t.replace(".", "-") for t in tickers]
        if len(result) > 400:
            return result
    except Exception:
        pass
    # Essai 2 : source alternative GitHub
    try:
        resp = requests.get(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
            timeout=20,
        )
        df = pd.read_csv(io.StringIO(resp.text))
        tickers = df["Symbol"].tolist()
        if len(tickers) > 400:
            return [t.replace(".", "-") for t in tickers]
    except Exception:
        pass
    logger.warning("SP500 fetch failed, using built-in fallback")
    return _SP500_FALLBACK


def get_nasdaq() -> list[str]:
    # Essai 1 : FTP via HTTP
    for url in [
        "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    ]:
        try:
            resp = requests.get(url, timeout=30, verify=False)
            df = pd.read_csv(io.StringIO(resp.text), sep="|")
            df = df[df["Test Issue"] == "N"]
            tickers = df["Symbol"].dropna().tolist()
            result = [t for t in tickers if isinstance(t, str) and t.replace("-", "").isalpha() and len(t) <= 5]
            if len(result) > 500:
                return result
        except Exception:
            continue
    logger.warning("NASDAQ fetch failed, using built-in fallback")
    return _NASDAQ_FALLBACK


def get_all_tickers() -> dict[str, list[str]]:
    return {
        "CAC40":  CAC40,
        "SBF120": SBF120,
        "SP500":  get_sp500(),
    }


# ── Fallbacks ─────────────────────────────────────────────────────────────────
_SP500_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "XOM", "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "AVGO", "COST", "MCD", "WMT", "BAC", "LLY",
    "TMO", "CRM", "ACN", "AMD", "NFLX", "ADBE", "TXN", "CSCO", "DHR",
    "NEE", "PM", "RTX", "QCOM", "UNP", "LIN", "BMY", "AMGN", "HON",
    "INTU", "SBUX", "IBM", "GE", "CAT",
]

_NASDAQ_FALLBACK = [
    # Mega-cap tech
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "ADBE", "QCOM", "TXN", "INTU", "AMAT", "MU", "LRCX", "SNPS",
    "CDNS", "KLAC", "MRVL", "FTNT", "ABNB", "TEAM", "WDAY", "CRWD", "ZS", "DDOG",
    "NET", "SNOW", "PLTR", "SMCI", "ARM", "SHOP", "PYPL", "OKTA", "PANW", "FAST",
    # Semiconducteurs
    "INTC", "MCHP", "ADI", "NXPI", "ON", "MPWR", "SWKS", "ENPH", "FSLR", "WOLF",
    # Biotech / Pharma
    "GILD", "AMGN", "BIIB", "REGN", "VRTX", "ILMN", "IDXX", "ALGN", "MRNA", "BNTX",
    "ALNY", "BMRN", "INCY", "IONS", "EXAS", "HOLX", "ISRG", "DXCM", "SGEN", "RARE",
    # Fintech / Finance
    "COIN", "HOOD", "MSTR", "NDAQ", "IBKR", "SCHW", "LPLA", "MKTX", "ETSY", "EBAY",
    # Cloud / Software
    "NOW", "VEEV", "HUBS", "DOCU", "ZM", "TWLO", "MDB", "ESTC", "CFLT", "GTLB",
    "PCTY", "PAYC", "ASAN", "BILL", "FIVN", "NEWR", "APPN", "DOMO", "BRZE", "TTD",
    # Consumer / Media
    "ROKU", "SPOT", "SNAP", "PINS", "MTCH", "CHTR", "LULU", "ROST", "ULTA", "FIVE",
    "DLTR", "BKNG", "EXPE", "VRSK", "CPRT", "ORLY", "PAYX", "CTAS", "ODFL", "LSTR",
    # EV & Mobility
    "RIVN", "LCID", "NIO", "XPEV", "LI", "DKNG", "PENN", "SPCE", "RKLB", "ACHR",
    # Healthcare & Devices
    "PODD", "TMDX", "ICLR", "MEDP", "NTRA", "NVST", "OMCL", "PRCT", "AXNX", "LIVN",
    # Enterprise & Infra
    "PSTG", "NTAP", "WDC", "STX", "KEYS", "ANSS", "CTSH", "EPAM", "GLOB", "TTWO",
    # Divers
    "FLEX", "JBHT", "CHRW", "EXPD", "FWRD", "WERN", "HUBG", "ARCB", "SAIA", "ECHO",
]
