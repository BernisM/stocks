from __future__ import annotations
"""
Fetches and caches all ticker lists: CAC40, SBF120, S&P500, NASDAQ.
"""
import io
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ── CAC 40 ────────────────────────────────────────────────────────────────────
CAC40 = [
    "AI.PA", "AIR.PA", "ALO.PA", "ACA.PA", "CS.PA", "BNP.PA", "EN.PA",
    "CAP.PA", "CA.PA", "SGO.PA", "ML.PA", "DSY.PA", "EDEN.PA", "ENGI.PA",
    "EL.PA", "ERF.PA", "RMS.PA", "KER.PA", "LR.PA", "OR.PA", "MC.PA",
    "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SAN.PA", "SU.PA",
    "GLE.PA", "STM.PA", "TEP.PA", "HO.PA", "TTE.PA", "URW.PA", "VIE.PA",
    "DG.PA", "VIV.PA", "BN.PA", "NK.PA", "WLN.PA",
]

# ── SBF 120 (CAC40 + 80 autres valeurs françaises) ────────────────────────────
_SBF120_EXTRA = [
    "AC.PA", "ATE.PA", "AMUN.PA", "AKE.PA", "BB.PA", "BIM.PA", "BVI.PA",
    "COV.PA", "COFA.PA", "FGR.PA", "RF.PA", "FNAC.PA", "GTT.PA", "GET.PA",
    "GFC.PA", "ICAD.PA", "IPN.PA", "DEC.PA", "MMB.PA", "NEX.PA", "NXI.PA",
    "POM.PA", "RCO.PA", "RUI.PA", "DIM.PA", "SCR.PA", "SK.PA", "SW.PA",
    "SOP.PA", "SPIE.PA", "TE.PA", "TKO.PA", "TFI.PA", "TRI.PA", "UBI.PA",
    "VK.PA", "VRLA.PA", "VIRB.PA", "MF.PA", "ADP.PA", "ALSTOM.PA", "APAM.PA",
    "BALO.PA", "BIGB.PA", "CBP.PA", "CGG.PA", "CHSR.PA", "CORA.PA", "CRI.PA",
    "DBV.PA", "DKUSY.PA", "ELIS.PA", "ERA.PA", "ESSO.PA", "EUCAR.PA",
    "FDE.PA", "FGEAL.PA", "FOUG.PA", "GBB.PA", "GBLB.PA", "HCO.PA", "ILD.PA",
    "IMCD.PA", "INFE.PA", "INUI.PA", "JACQ.PA", "KORI.PA", "LACR.PA",
    "LOUP.PA", "LSTR.PA", "MDM.PA", "MERY.PA", "MIDI.PA", "MKGP.PA",
    "MLFP.PA", "MLNF.PA", "MROS.PA", "NACON.PA", "NEO.PA", "ORIA.PA",
    "PCAS.PA", "POXEL.PA",
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
    # Essai 2 : source alternative
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
    # Essai 1 : FTP via HTTP (pas HTTPS)
    for url in [
        "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    ]:
        try:
            resp = requests.get(url, timeout=30, verify=False)
            df = pd.read_csv(io.StringIO(resp.text), sep="|")
            df = df[df["Test Issue"] == "N"]
            tickers = df["Symbol"].dropna().tolist()
            result = [t for t in tickers if isinstance(t, str) and t.replace("-","").isalpha() and len(t) <= 5]
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
        "NASDAQ": get_nasdaq(),
    }


# ── Fallbacks (top 50 par marché si le réseau échoue) ─────────────────────────
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
    "CDNS", "KLAC", "MRVL", "FTNT", "ASML", "ABNB", "DXCM", "TEAM", "WDAY", "CRWD",
    "ZS", "OKTA", "DDOG", "NET", "SNOW", "PLTR", "SMCI", "ARM", "SHOP", "PYPL",
    # Semiconducteurs
    "INTC", "MCHP", "ADI", "XLNX", "NXPI", "ON", "MPWR", "SWKS", "QRVO", "WOLF",
    "ENPH", "SEDG", "FSLR", "CSIQ", "SPWR",
    # Biotech / Pharma
    "GILD", "AMGN", "BIIB", "REGN", "VRTX", "ILMN", "IDXX", "ALGN", "DXCM", "HOLX",
    "MRNA", "BNTX", "NVAX", "SGEN", "ALNY", "BMRN", "INCY", "EXAS", "IONS", "FATE",
    # Fintech / Finance
    "COIN", "SQ", "SOFI", "HOOD", "MSTR", "AFRM", "UPST", "LC", "OPEN", "RELY",
    "NDAQ", "MKTX", "LPLA", "IBKR", "SCHW", "ETSY", "EBAY", "CPNG", "SE", "GRAB",
    # Cloud / Software
    "CRM", "NOW", "VEEV", "HUBS", "DOCU", "ZM", "TWLO", "FIVN", "NEWR", "SPLK",
    "MDB", "ESTC", "CFLT", "GTLB", "DOMO", "APPN", "PCTY", "PAYC", "ASAN", "BILL",
    # Consumer / Media
    "NFLX", "ROKU", "SPOT", "SNAP", "PINS", "MTCH", "IAC", "WBD", "PARA", "FOX",
    "FOXA", "CHTR", "LULU", "ROST", "ULTA", "FIVE", "DLTR", "BKNG", "EXPE", "TRIP",
    # Divers
    "RIVN", "LCID", "NIO", "XPEV", "LI", "FFIE", "GOEV", "FSR", "WKHS", "SOLO",
    "HOOD", "OPEN", "DKNG", "PENN", "CZOO", "SPCE", "RKLB", "ASTR", "MNTS", "RDW",
]
