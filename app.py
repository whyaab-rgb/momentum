from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BSJP SNIPER IDX", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# GET ALL IDX STOCK
# =========================================================
def get_all_idx_symbols():
    try:
        tables = pd.read_html("https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/")
        symbols = set()

        for table in tables:
            cols = [str(c).upper() for c in table.columns]
            table.columns = cols

            for col in ["KODE SAHAM", "CODE", "STOCK CODE"]:
                if col in table.columns:
                    for val in table[col].dropna():
                        val = str(val).strip().upper()
                        if val.isalnum():
                            symbols.add(val + ".JK")

        return list(symbols)
    except:
        return ["BBRI.JK","BMRI.JK","TLKM.JK","BRIS.JK"]

# =========================================================
# HELPERS
# =========================================================
def latest(series):
    try:
        return float(series.iloc[-1])
    except:
        return 0

# =========================================================
# INDICATORS
# =========================================================
def calc(df):
    x = df.copy()

    x["EMA9"] = x["Close"].ewm(span=9).mean()
    x["EMA20"] = x["Close"].ewm(span=20).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA50"] = x["Close"].rolling(50).mean()
    x["VOL_MA20"] = x["Volume"].rolling(20).mean()

    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, 1)
    x["RSI"] = 100 - (100 / (1 + rs))

    ema12 = x["Close"].ewm(span=12).mean()
    ema26 = x["Close"].ewm(span=26).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9).mean()

    x["SUPPORT"] = x["Low"].rolling(20).min()
    x["RESIST"] = x["High"].rolling(20).max()

    return x

# =========================================================
# SNIPER LOGIC
# =========================================================
def sniper(close, ema9, ema20, rsi, macd, macd_sig, vol, vol_ma):
    if close > ema9 > ema20 and macd > macd_sig and 55 <= rsi <= 70 and vol > vol_ma:
        return "SNIPER ENTRY 🔥"
    if rsi < 35:
        return "REBOUND ⚡"
    if rsi > 75:
        return "OVERBOUGHT ⚠️"
    return "WAIT"

def decision(now, entry, tp, sl):
    if now <= sl:
        return "JUAL ❌"
    if now >= tp:
        return "TAKE PROFIT 💰"
    if now <= entry * 1.02:
        return "MASUK 🔥"
    return "PANTAU 👀"

# =========================================================
# DATA DOWNLOAD
# =========================================================
def get_data(symbol):
    df = yf.download(symbol, period="3mo", interval="1d", progress=False)
    if df.empty:
        return None
    return calc(df)

# =========================================================
# MAIN ENDPOINT
# =========================================================
@app.get("/bsjp")
def scan(symbols: Optional[str] = None):

    if symbols:
        symbol_list = [s.strip().upper() + ".JK" for s in symbols.split(",")]
    else:
        symbol_list = get_all_idx_symbols()

    rows = []

    for s in symbol_list[:150]:  # limit biar gak berat
        try:
            df = get_data(s)
            if df is None or len(df) < 30:
                continue

            close = latest(df["Close"])
            prev = df["Close"].iloc[-2]

            gain = (close - prev) / prev * 100
            rsi = latest(df["RSI"])
            macd = latest(df["MACD"])
            macd_sig = latest(df["MACD_SIGNAL"])
            vol = latest(df["Volume"])
            vol_ma = latest(df["VOL_MA20"])
            ema9 = latest(df["EMA9"])
            ema20 = latest(df["EMA20"])

            support = latest(df["SUPPORT"])
            entry = (support + ema20) / 2 if support else close
            tp = close * 1.05
            sl = close * 0.97

            sig = sniper(close, ema9, ema20, rsi, macd, macd_sig, vol, vol_ma)
            dec = decision(close, entry, tp, sl)

            rows.append({
                "ticker": s.replace(".JK",""),
                "price": round(close,2),
                "gain": round(gain,2),
                "rsi": round(rsi,1),
                "rvol": round(vol/vol_ma*100,1) if vol_ma else 0,
                "signal": sig,
                "decision": dec,
                "entry": round(entry,2),
                "tp": round(tp,2),
                "sl": round(sl,2)
            })

        except:
            continue

    rows = sorted(rows, key=lambda x: x["gain"], reverse=True)

    return {
        "data": rows,
        "total": len(rows),
        "time": datetime.now().isoformat()
    }

# =========================================================
# CANDLE ENDPOINT
# =========================================================
@app.get("/candles")
def candles(symbol: str):

    symbol = symbol.upper() + ".JK"

    df = yf.download(symbol, period="5d", interval="5m", progress=False)

    if df.empty:
        return {"data":[]}

    df = calc(df)

    out = []

    for i in range(len(df)):
        row = df.iloc[i]

        buy = row["Close"] > row["EMA9"] > row["EMA20"]
        sell = row["RSI"] > 75

        out.append({
            "time": str(df.index[i]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "buy": bool(buy),
            "sell": bool(sell)
        })

    return {"data": out}
