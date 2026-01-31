TICKER_SETS = {
    "WIG20": {
        "benchmark": "^WIG20",
        "tickers": [
            "PKN.WA",  # Orlen
            "PKO.WA",  # PKO BP
            "PZU.WA",  # PZU
            "PEO.WA",  # Bank Pekao
            "KGH.WA",  # KGHM
            "CDR.WA",  # CD Projekt
            "DNP.WA",  # Dino Polska
            "LPP.WA",  # LPP
            "ALE.WA",  # Allegro
            "CPS.WA",  # Cyfrowy Polsat
            "OPL.WA",  # Orange Polska
            "JSW.WA",  # JSW
            "CCC.WA",  # CCC
            "MBK.WA",  # mBank
            "SPL.WA",  # Santander Bank Polska
            "PGE.WA",  # PGE
            "TPE.WA",  # Tauron
            "KTY.WA",  # Kety
            "ACP.WA",  # Asseco Poland
            "LWB.WA",  # Bogdanka
        ],
    },
    "WIG_BROAD": {
        "benchmark": "GPW.WA",
        "tickers": [
            # Banks & Finance
            "ING.WA",
            "MIL.WA",
            "ALR.WA",
            "KRU.WA",
            "XTB.WA",
            "GPW.WA",
            "HRP.WA",
            # Energy & Industry (some delisted included for attempt)
            "ENA.WA",
            "BDX.WA",
            "ATT.WA",
            "STP.WA",
            # Consumer & Tech
            "TEN.WA",
            "11B.WA",
            "TXT.WA",
            "EUR.WA",
            "ASB.WA",
            "NEU.WA",
            # Construction & Real Estate
            "DOM.WA",
            "ECH.WA",
            "DVL.WA",
        ],
    },
    "US_TECH": {
        "benchmark": "^NDX",
        "tickers": [
            "AAPL",  # Apple
            "MSFT",  # Microsoft
            "GOOGL",  # Alphabet (Google)
            "AMZN",  # Amazon
            "NVDA",  # Nvidia
            "META",  # Meta (Facebook)
            "TSLA",  # Tesla
            "AMD",  # Advanced Micro Devices
            "INTC",  # Intel
            "CRM",  # Salesforce
        ],
    },
    "US_DEFENSIVE": {
        "benchmark": "^GSPC",
        "tickers": [
            "KO",  # Coca-Cola
            "PEP",  # PepsiCo
            "PG",  # Procter & Gamble
            "JNJ",  # Johnson & Johnson
            "MCD",  # McDonald's
            "WMT",  # Walmart
            "VZ",  # Verizon
            "MRK",  # Merck
            "PFE",  # Pfizer
            "COST",  # Costco
        ],
    },
    "ETFS": {
        "benchmark": "ACWI",
        "tickers": [
            "SPY",  # S&P 500
            "QQQ",  # Nasdaq 100
            "EEM",  # Emerging Markets
            "EFA",  # EAFE (Europe, Australia, Asia, Far East)
            "AGG",  # US Aggregate Bond
            "GLD",  # Gold
            "IWM",  # Russell 2000 (Small Cap)
        ],
    },
    "CRYPTO": {
        "benchmark": "BTC-USD",
        "tickers": [
            "BTC-USD",  # Bitcoin
            "ETH-USD",  # Ethereum
            "SOL-USD",  # Solana
            "BNB-USD",  # Binance Coin
            "ADA-USD",  # Cardano
            "XRP-USD",  # XRP
        ],
    },
}

# Dynamically add WIG20 tickers to WIG_BROAD to avoid redundancy
TICKER_SETS["WIG_BROAD"]["tickers"] = TICKER_SETS["WIG20"]["tickers"] + TICKER_SETS["WIG_BROAD"]["tickers"]

DEFAULT_TICKER_SET = "WIG20"
