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
    "RNO.PA",  "SAF.PA",  "SAN.PA",  "SU.PA",   "GLE.PA",
    "HO.PA",   "TTE.PA",  "URW.PA",  "VIE.PA",  "DG.PA",
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
    # "KLE.PA",  # Klépierre — delisted 2026-04
    "MF.PA",    # Wendel
    "RF.PA",    # Eurazeo
    # "SCOR.PA", # SCOR — delisted 2026-04
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
    # "ILD.PA",   # Iliad (Free) — delisted/unavailable 2026-04
    "MMB.PA",   # Lagardère
    "SOP.PA",   # Sopra Steria
    "SW.PA",    # Sodexo
    "TE.PA",    # Technip Energies
    "TFI.PA",   # TF1
    "TRI.PA",   # Trigano
    # "SII.PA",   # SII Group — delisted 2026-04
    # Pharma & Santé
    # "BIOA.PA",  # bioMérieux — delisted/unavailable 2026-04
    "IPN.PA",   # Ipsen
    # "VIRB.PA",  # Virbac — delisted/unavailable 2026-04
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



# ── Matières premières (contrats futures continus yfinance) ──────────────────
COMMODITY_NAMES: dict[str, str] = {
    "GC=F":  "Or (Gold)",
    "SI=F":  "Argent (Silver)",
    "CL=F":  "Pétrole WTI",
    "BZ=F":  "Pétrole Brent",
    "NG=F":  "Gaz Naturel",
    "HG=F":  "Cuivre (Copper)",
    "PL=F":  "Platine",
    "PA=F":  "Palladium",
    "ZW=F":  "Blé (Wheat)",
    "ZC=F":  "Maïs (Corn)",
    "KC=F":  "Café (Coffee)",
    "SB=F":  "Sucre #11 (Sugar)",
    "CC=F":  "Cacao (Cocoa)",
}
COMMODITIES: list[str] = list(COMMODITY_NAMES.keys())


# ── NASDAQ Growth — small/mid caps US ciblées "futures licornes" ──────────────
# Liste curatée : tech disruptive, biotech, EV, fintech, espace, IA, quantum.
# Volontairement hors mega caps déjà couvertes par SP500.
NASDAQ_GROWTH: list[str] = [
    # Espace / Défense / Aérospatial
    "ASTS", "RKLB", "ACHR", "JOBY", "SPCE", "KTOS", "LUNR", "RDW",
    # IA / Quantum / Compute
    "AI", "SOUN", "BBAI", "IONQ", "RGTI", "QBTS", "ARQQ", "PATH", "SMCI", "ARM",
    # Fintech / Crypto
    "SOFI", "UPST", "AFRM", "COIN", "HOOD", "MARA", "RIOT", "CLSK", "MSTR", "BITF", "HUT",
    # EV / Mobilité / Cleantech
    "RIVN", "LCID", "NIO", "XPEV", "LI", "PLUG", "FCEL", "BE", "RUN", "CHPT", "BLNK", "QS",
    # Solaire / Hydrogène
    "ENPH", "FSLR", "SEDG", "ARRY", "BLDP",
    # Biotech / Génomique / Thérapie génique
    "VKTX", "GERN", "SRPT", "MRNA", "BNTX", "RXRX", "CRSP", "EDIT", "NTLA", "PRME", "BEAM",
    "PACB", "TWST", "SGMO", "ALNY",
    # Cloud / Cyber / SaaS
    "PLTR", "CRWD", "ZS", "DDOG", "S", "OKTA", "PANW", "SNOW", "MDB", "NET", "GTLB",
    "ESTC", "CFLT", "BILL", "BRZE", "ASAN", "PCTY", "TWLO", "FROG",
    # Consumer Tech / Media
    "RBLX", "U", "ROKU", "SPOT", "PINS", "SNAP", "DUOL", "RDDT",
    # E-commerce / Marketplace
    "SHOP", "MELI", "SE", "FVRR", "ETSY", "JMIA", "GLBE",
    # Gaming / Paris sportifs
    "DKNG", "PENN", "GLPI", "FUBO", "ABNB",
    # Healthcare devices / MedTech
    "DXCM", "INMD", "IRTC", "PEN", "OMCL", "TMDX",
    # SaaS / Vertical software
    "VEEV", "HUBS", "DOCU", "ZM", "TTD", "APP", "GLOB", "EPAM",
    # Divers tech / disrupteurs
    "DASH", "UBER", "LYFT", "RBLX", "PATH", "CART", "RUM",
]
# Déduplication
NASDAQ_GROWTH = list(dict.fromkeys(NASDAQ_GROWTH))


# ── Euronext Growth — small caps françaises ───────────────────────────────────
# Liste curatée de PME en croissance sur Euronext Growth Paris (préfixe ALxxxx).
# Volontairement restreinte aux tickers avec couverture yfinance fiable.
EURONEXT_GROWTH: list[str] = [
    # Vérifiés actifs sur yfinance (avril 2026)
    "ALCJ.PA",   # Crossject
    "ALEUP.PA",  # Europlasma
    "ALMER.PA",  # Mercialys
    "ALENT.PA",  # Entech
    "ALCRB.PA",  # Carbios (chimie verte)
    "ALSPW.PA",  # Spineway
    "ALTHO.PA",  # Théradiag
    "ALORD.PA",  # Ordissimo
    "ALNXT.PA",  # Nextstage AM
    "ALAFY.PA",  # Affluent Medical
    "ALBOO.PA",  # Boostheat
    "ALBPS.PA",  # Biophytis
    "ALDLS.PA",  # Delfingen Industry
]
# Déduplication
EURONEXT_GROWTH = list(dict.fromkeys(EURONEXT_GROWTH))


# ── Cryptomonnaies (paires USD yfinance) ──────────────────────────────────────
CRYPTO_NAMES: dict[str, str] = {
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "BNB-USD":  "BNB",
    "XRP-USD":  "XRP",
    "SOL-USD":  "Solana",
    "ADA-USD":  "Cardano",
    "DOGE-USD": "Dogecoin",
    "AVAX-USD": "Avalanche",
    "DOT-USD":  "Polkadot",
    "LINK-USD": "Chainlink",
    "LTC-USD":  "Litecoin",
    "BCH-USD":  "Bitcoin Cash",
    # "UNI-USD":  "Uniswap",  — unavailable on yfinance 2026-04
    "ATOM-USD": "Cosmos",
    "XLM-USD":  "Stellar",
}
CRYPTOS: list[str] = list(CRYPTO_NAMES.keys())


def get_all_tickers() -> dict[str, list[str]]:
    """
    Retourne tous les tickers par marché. Utilise le cache de refresh hebdo
    si disponible, sinon les listes hardcodées (fallback).
    """
    from .ticker_refresh import get_cached_list

    cac40  = get_cached_list("CAC40")  or CAC40
    sbf120 = get_cached_list("SBF120") or SBF120
    sp500  = get_sp500()

    nasdaq_dynamic = get_cached_list("NASDAQ_GROWTH")
    nasdaq_base    = nasdaq_dynamic if nasdaq_dynamic else NASDAQ_GROWTH
    nasdaq         = [t for t in nasdaq_base if t not in sp500]   # éviter doublons

    euronext_dynamic = get_cached_list("EURONEXT_GROWTH")
    euronext         = euronext_dynamic if euronext_dynamic else EURONEXT_GROWTH

    return {
        "CAC40":           cac40,
        "SBF120":          sbf120,
        "EURONEXT_GROWTH": euronext,
        "SP500":           sp500,
        "NASDAQ":          nasdaq,
        "COMMODITIES":     COMMODITIES,
        "CRYPTO":          CRYPTOS,
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
