import math
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IDX Momentum Premium Screener",
    layout="wide"
)

# =========================================================
# CONFIG
# =========================================================
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"
CHUNK_SIZE = 80

# =========================================================
# UI STYLE
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 14px;
}
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
    max-width: 100%;
}
h1, h2, h3 {
    margin-top: 0.2rem !important;
    margin-bottom: 0.4rem !important;
}
div[data-testid="stMetric"] {
    background: #0b1220;
    border: 1px solid #1f2937;
    padding: 10px 14px;
    border-radius: 12px;
}
div[data-testid="stMetricLabel"] {
    color: #cbd5e1;
}
div[data-testid="stMetricValue"] {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🔥 IDX Momentum Premium Screener")
st.caption("Auto scan semua saham IDX dengan tabel warna, ranking score premium, dan probabilitas entry.")

# =========================================================
# HELPERS
# =========================================================
def clean_idx_code(code: str):
    if pd.isna(code):
        return None
    code = str(code).strip().upper()
    if not code:
        return None
    if "-" in code:
        code = code.split("-")[0].strip()
    code = "".join(ch for ch in code if ch.isalnum())
    if len(code) < 4:
        return None
    return code

@st.cache_data(ttl=3600)
def get_idx_symbols():
    symbols = set()

    # Coba ambil dari IDX
    try:
        url = "https://www.idx.co.id/en/market-data/stocks-data/stock-list"
        tables = pd.read_html(url)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            for candidate in ["code", "stock code", "ticker", "symbol"]:
                if candidate in cols:
                    col = t.columns[cols.index(candidate)]
                    vals = t[col].astype(str).tolist()
                    for v in vals:
                        c = clean_idx_code(v)
                        if c:
                            symbols.add(c)
        if symbols:
            return sorted(symbols)
    except:
        pass

    # Fallback
    try:
        url = "https://stockanalysis.com/list/indonesia-stock-exchange/"
        tables = pd.read_html(url)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            for candidate in ["symbol", "ticker"]:
                if candidate in cols:
                    col = t.columns[cols.index(candidate)]
                    vals = t[col].astype(str).tolist()
                    for v in vals:
                        c = clean_idx_code(v)
                        if c:
                            symbols.add(c)
        if symbols:
            return sorted(symbols)
    except:
        pass

    # fallback manual
    fallback = [
        "BBCA","BBRI","BMRI","BBNI","ASII","TLKM","ICBP","INDF","CPIN","ADRO",
        "MDKA","ANTM","UNTR","BRPT","AMMN","GOTO","BUKA","PGAS","EXCL","AKRA",
        "SMGR","KLBF","MAPI","ACES","ERAA","JPFA","PTBA","ITMG","TPIA","JSMR",
        "SIDO","SMRA","CTRA","PWON","BSDE","ESSA","MEDC","AADI","BREN","AMRT",
        "MTDL","BJBR","SMDR","BBYB","ARTO","BRIS","ENRG","HRUM","PTRO","INDY"
    ]
    return sorted(set(fallback))

def yahoo_symbol(code: str) -> str:
    return f"{code}.JK"

def calc_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_wick_pct(high, low, close, open_):
    rng = high - low
    if rng <= 0:
        return 0.0
    upper_wick = high - max(open_, close)
    return max(0.0, (upper_wick / rng) * 100)

def calc_val_display(close_price, volume):
    value = close_price * volume
    if value >= 1_000_000_000_000:
        return f"{value/1_000_000_000_000:.1f}T"
    elif value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    return f"{value:,.0f}"

# =========================================================
# LABELS
# =========================================================
def trend_label(close, sma20, sma50):
    if close > sma20 > sma50:
        return "BULL"
    elif close > sma20:
        return "NEUTRAL"
    return "BEAR"

def rsi_signal(rsi):
    if rsi >= 55:
        return "UP"
    elif rsi >= 45:
        return "SIDE"
    return "DOWN"

def phase_label(close, sma20, sma50, rvol):
    if close > sma20 and sma20 > sma50 and rvol >= 180:
        return "BIG AKUM"
    elif close > sma20 and rvol >= 110:
        return "AKUM"
    elif close < sma20 and rvol < 90:
        return "DISTRIB"
    return "NEUTRAL"

def signal_label(close, high20, sma20, rsi, rvol, wick):
    if close >= high20 * 0.995 and rsi >= 58 and rvol >= 140 and wick <= 35:
        return "SUPER"
    elif close > sma20 and rsi >= 52 and rvol >= 110:
        return "AKUM"
    elif close >= high20 * 0.97 and rsi >= 48:
        return "ON TRACK"
    elif close <= sma20 * 1.01:
        return "AT ENTRY"
    return "GC NOW"

def action_label(score, now_price, entry, trend):
    diff = ((now_price - entry) / entry) * 100 if entry > 0 else 999

    if score >= 85 and trend == "BULL" and diff <= 2:
        return "SIAP BELI"
    elif score >= 75 and diff <= 4:
        return "WATCH"
    elif diff <= 1.2:
        return "AT ENTRY"
    return "WAIT"

# =========================================================
# SCORE
# =========================================================
def score_gain(gain):
    if gain >= 5:
        return 100
    elif gain >= 3:
        return 92
    elif gain >= 2:
        return 84
    elif gain >= 1:
        return 74
    elif gain >= 0:
        return 60
    elif gain >= -2:
        return 40
    return 20

def score_rvol(rvol):
    if rvol >= 250:
        return 100
    elif rvol >= 180:
        return 92
    elif rvol >= 140:
        return 84
    elif rvol >= 110:
        return 72
    elif rvol >= 95:
        return 58
    return 28

def score_rsi(rsi):
    if 56 <= rsi <= 68:
        return 100
    elif 52 <= rsi < 56:
        return 88
    elif 68 < rsi <= 75:
        return 76
    elif 48 <= rsi < 52:
        return 64
    elif 42 <= rsi < 48:
        return 48
    return 25

def score_trend(trend):
    return {
        "BULL": 100,
        "NEUTRAL": 62,
        "BEAR": 18
    }.get(trend, 50)

def score_wick(wick):
    if wick <= 15:
        return 100
    elif wick <= 30:
        return 88
    elif wick <= 45:
        return 72
    elif wick <= 65:
        return 52
    return 22

def score_entry(entry, now_):
    if entry <= 0 or now_ <= 0:
        return 20
    diff = ((now_ - entry) / entry) * 100
    if diff <= 0.8:
        return 100
    elif diff <= 2:
        return 90
    elif diff <= 4:
        return 74
    elif diff <= 6:
        return 55
    elif diff <= 9:
        return 35
    return 18

def score_signal(signal, phase):
    s1 = {
        "SUPER": 100,
        "AKUM": 82,
        "ON TRACK": 68,
        "AT ENTRY": 74,
        "GC NOW": 55
    }.get(signal, 45)

    s2 = {
        "BIG AKUM": 100,
        "AKUM": 82,
        "NEUTRAL": 56,
        "DISTRIB": 20
    }.get(phase, 50)

    return round((s1 * 0.6) + (s2 * 0.4), 2)

def premium_score(row):
    score = (
        score_gain(row["GAIN"]) * 0.14 +
        score_rvol(row["RVOL"]) * 0.20 +
        score_rsi(row["RSI 5M"]) * 0.14 +
        score_trend(row["TREND"]) * 0.14 +
        score_wick(row["WICK"]) * 0.08 +
        score_entry(row["ENTRY"], row["NOW"]) * 0.12 +
        score_signal(row["SINYAL"], row["FASE"]) * 0.10 +
        min(row["PROFIT"], 15) / 15 * 100 * 0.08
    )
    return round(score, 2)

def probability_label(score):
    if score >= 85:
        return "HIGH"
    elif score >= 72:
        return "MEDIUM"
    return "LOW"

def rank_badge(score):
    if score >= 90:
        return "A+"
    elif score >= 85:
        return "A"
    elif score >= 80:
        return "B+"
    elif score >= 75:
        return "B"
    elif score >= 68:
        return "C+"
    elif score >= 60:
        return "C"
    return "D"

# =========================================================
# DOWNLOAD DATA
# =========================================================
def download_chunk(symbols_jk, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    try:
        data = yf.download(
            tickers=symbols_jk,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
        return data
    except:
        return pd.DataFrame()

def extract_one_symbol_df(raw, sym_jk):
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        if sym_jk not in raw.columns.get_level_values(0):
            return None
        df = raw[sym_jk].copy()
    else:
        df = raw.copy()

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            return None

    df = df.dropna(subset=needed)
    if df.empty or len(df) < 60:
        return None

    return df

def analyze_symbol(code, df):
    try:
        close = df["Close"].astype(float)
        open_ = df["Open"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float).fillna(0)

        now_ = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else now_
        gain = ((now_ - prev) / prev) * 100 if prev > 0 else 0.0

        wick = calc_wick_pct(
            float(high.iloc[-1]),
            float(low.iloc[-1]),
            float(close.iloc[-1]),
            float(open_.iloc[-1])
        )

        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        high20 = float(high.tail(20).max())

        rsi = float(calc_rsi(close, 14).iloc[-1])
        rvol_base = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
        rvol = (float(volume.iloc[-1]) / rvol_base) * 100 if rvol_base > 0 else 0.0

        trend = trend_label(now_, sma20, sma50)
        rsi_sig = rsi_signal(rsi)
        fase = phase_label(now_, sma20, sma50, rvol)
        sinyal = signal_label(now_, high20, sma20, rsi, rvol, wick)

        atr = float((high - low).rolling(14).mean().iloc[-1]) if len(df) >= 14 else float((high - low).mean())
        entry = round(max(sma20, high.tail(5).max() * 0.985), 2)
        tp = round(now_ + (atr * 1.6), 2) if not math.isnan(atr) else round(now_ * 1.05, 2)
        sl = round(max(now_ - (atr * 1.1), sma20 * 0.97), 2) if not math.isnan(atr) else round(now_ * 0.97, 2)

        profit = ((tp - now_) / now_) * 100 if now_ > 0 else 0.0
        to_tp = ((tp - entry) / entry) * 100 if entry > 0 else 0.0
        val = calc_val_display(now_, float(volume.iloc[-1]))

        row = {
            "EMITEN": code,
            "GAIN": round(gain, 1),
            "WICK": round(wick, 1),
            "SINYAL": sinyal,
            "RVOL": round(rvol, 1),
            "ENTRY": round(entry, 2),
            "NOW": round(now_, 2),
            "TP": round(tp, 2),
            "SL": round(sl, 2),
            "PROFIT": round(profit, 1),
            "%TO TP": round(to_tp, 1),
            "RSI SIG": rsi_sig,
            "RSI 5M": round(rsi, 1),
            "VAL": val,
            "FASE": fase,
            "TREND": trend,
        }

        row["SCORE"] = premium_score(row)
        row["RANK"] = rank_badge(row["SCORE"])
        row["AKSI"] = action_label(row["SCORE"], row["NOW"], row["ENTRY"], row["TREND"])
        row["PROB"] = probability_label(row["SCORE"])

        return row
    except:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def run_scan(symbols, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    rows = []
    symbols_jk = [yahoo_symbol(s) for s in symbols]
    chunks = [symbols_jk[i:i+CHUNK_SIZE] for i in range(0, len(symbols_jk), CHUNK_SIZE)]

    for chunk in chunks:
        raw = download_chunk(chunk, period=period, interval=interval)
        for sym_jk in chunk:
            code = sym_jk.replace(".JK", "")
            df = extract_one_symbol_df(raw, sym_jk)
            if df is None:
                continue
            row = analyze_symbol(code, df)
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    out = out.sort_values(
        by=["SCORE", "RVOL", "GAIN"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    out.insert(0, "NO", range(1, len(out) + 1))

    cols = [
        "NO", "EMITEN", "GAIN", "WICK", "AKSI", "SINYAL", "RVOL",
        "ENTRY", "NOW", "TP", "SL", "PROFIT", "%TO TP",
        "RSI SIG", "RSI 5M", "VAL", "FASE", "TREND", "SCORE", "RANK", "PROB"
    ]
    return out[cols]

# =========================================================
# COLOR STYLE
# =========================================================
def style_base(_):
    return "background-color:#123b6d; color:white; font-weight:bold; text-align:center;"

def style_emiten(_):
    return "background-color:#1d4ed8; color:white; font-weight:bold; text-align:center;"

def style_gain(val):
    try:
        v = float(val)
        if v >= 2:
            return "background-color:#16a34a; color:white; font-weight:bold; text-align:center;"
        elif v > 0:
            return "background-color:#22c55e; color:white; font-weight:bold; text-align:center;"
        elif v == 0:
            return "background-color:#ef4444; color:white; font-weight:bold; text-align:center;"
        return "background-color:#dc2626; color:white; font-weight:bold; text-align:center;"
    except:
        return ""

def style_wick(val):
    try:
        v = float(val)
        if v <= 25:
            return "background-color:#16a34a; color:white; font-weight:bold; text-align:center;"
        elif v <= 45:
            return "background-color:#f59e0b; color:white; font-weight:bold; text-align:center;"
        return "background-color:#dc2626; color:white; font-weight:bold; text-align:center;"
    except:
        return ""

def style_aksi(val):
    mp = {
        "SIAP BELI": "background-color:#7e22ce; color:white; font-weight:bold; text-align:center;",
        "WATCH": "background-color:#374151; color:white; font-weight:bold; text-align:center;",
        "AT ENTRY": "background-color:#7c3aed; color:white; font-weight:bold; text-align:center;",
        "WAIT": "background-color:#475569; color:white; font-weight:bold; text-align:center;"
    }
    return mp.get(val, "text-align:center;")

def style_sinyal(val):
    mp = {
        "SUPER": "background-color:#6d28d9; color:white; font-weight:bold; text-align:center;",
        "AKUM": "background-color:#16a34a; color:white; font-weight:bold; text-align:center;",
        "ON TRACK": "background-color:#15803d; color:white; font-weight:bold; text-align:center;",
        "AT ENTRY": "background-color:#2563eb; color:white; font-weight:bold; text-align:center;",
        "GC NOW": "background-color:#7c3aed; color:white; font-weight:bold; text-align:center;"
    }
    return mp.get(val, "text-align:center;")

def style_rvol(val):
    try:
        v = float(val)
        if v >= 180:
            return "background-color:#f97316; color:white; font-weight:bold; text-align:center;"
        elif v >= 120:
            return "background-color:#fb923c; color:white; font-weight:bold; text-align:center;"
        return "background-color:#9ca3af; color:black; font-weight:bold; text-align:center;"
    except:
        return ""

def style_price(_):
    return "background-color:#2563eb; color:white; font-weight:bold; text-align:center;"

def style_tp(_):
    return "background-color:#16a34a; color:white; font-weight:bold; text-align:center;"

def style_sl(_):
    return "background-color:#dc2626; color:white; font-weight:bold; text-align:center;"

def style_profit(val):
    try:
        v = float(val)
        if v >= 8:
            return "background-color:#16a34a; color:white; font-weight:bold; text-align:center;"
        elif v >= 4:
            return "background-color:#22c55e; color:white; font-weight:bold; text-align:center;"
        elif v > 0:
            return "background-color:#84cc16; color:black; font-weight:bold; text-align:center;"
        return "background-color:#dc2626; color:white; font-weight:bold; text-align:center;"
    except:
        return ""

def style_to_tp(val):
    try:
        v = float(val)
        if v >= 8:
            return "background-color:#0f766e; color:white; font-weight:bold; text-align:center;"
        elif v >= 4:
            return "background-color:#0d9488; color:white; font-weight:bold; text-align:center;"
        return "background-color:#14b8a6; color:white; font-weight:bold; text-align:center;"
    except:
        return ""

def style_rsi_sig(val):
    mp = {
        "UP": "background-color:#16a34a; color:white; font-weight:bold; text-align:center;",
        "SIDE": "background-color:#7e22ce; color:white; font-weight:bold; text-align:center;",
        "DOWN": "background-color:#dc2626; color:white; font-weight:bold; text-align:center;"
    }
    return mp.get(val, "text-align:center;")

def style_rsi(val):
    try:
        v = float(val)
        if v >= 60:
            return "background-color:#2563eb; color:white; font-weight:bold; text-align:center;"
        elif v >= 50:
            return "background-color:#3b82f6; color:white; font-weight:bold; text-align:center;"
        elif v >= 40:
            return "background-color:#7c3aed; color:white; font-weight:bold; text-align:center;"
        return "background-color:#dc2626; color:white; font-weight:bold; text-align:center;"
    except:
        return ""

def style_val(_):
    return "background-color:#1f3a8a; color:white; font-weight:bold; text-align:center;"

def style_fase(val):
    mp = {
        "BIG AKUM": "background-color:#7c3aed; color:white; font-weight:bold; text-align:center;",
        "AKUM": "background-color:#16a34a; color:white; font-weight:bold; text-align:center;",
        "NEUTRAL": "background-color:#4b5563; color:white; font-weight:bold; text-align:center;",
        "DISTRIB": "background-color:#dc2626; color:white; font-weight:bold; text-align:center;"
    }
    return mp.get(val, "text-align:center;")

def style_trend(val):
    mp = {
        "BULL": "background-color:#16a34a; color:white; font-weight:bold; text-align:center;",
        "NEUTRAL": "background-color:#6b7280; color:white; font-weight:bold; text-align:center;",
        "BEAR": "background-color:#dc2626; color:white; font-weight:bold; text-align:center;"
    }
    return mp.get(val, "text-align:center;")

def style_score(val):
    try:
        v = float(val)
        if v >= 90:
            return "background-color:#f59e0b; color:black; font-weight:900; text-align:center;"
        elif v >= 85:
            return "background-color:#facc15; color:black; font-weight:900; text-align:center;"
        elif v >= 75:
            return "background-color:#22c55e; color:white; font-weight:900; text-align:center;"
        elif v >= 65:
            return "background-color:#0ea5e9; color:white; font-weight:900; text-align:center;"
        return "background-color:#6b7280; color:white; font-weight:900; text-align:center;"
    except:
        return ""

def style_rank(val):
    mp = {
        "A+": "background-color:#f59e0b; color:black; font-weight:900; text-align:center;",
        "A": "background-color:#facc15; color:black; font-weight:900; text-align:center;",
        "B+": "background-color:#22c55e; color:white; font-weight:900; text-align:center;",
        "B": "background-color:#0ea5e9; color:white; font-weight:900; text-align:center;",
        "C+": "background-color:#6366f1; color:white; font-weight:900; text-align:center;",
        "C": "background-color:#6b7280; color:white; font-weight:900; text-align:center;",
        "D": "background-color:#111827; color:white; font-weight:900; text-align:center;"
    }
    return mp.get(val, "text-align:center;")

def style_prob(val):
    mp = {
        "HIGH": "background-color:#16a34a; color:white; font-weight:900; text-align:center;",
        "MEDIUM": "background-color:#f97316; color:white; font-weight:900; text-align:center;",
        "LOW": "background-color:#dc2626; color:white; font-weight:900; text-align:center;"
    }
    return mp.get(val, "text-align:center;")

def style_dataframe(df):
    return (
        df.style
        .format({
            "GAIN": "{:.1f}%",
            "WICK": "{:.1f}%",
            "RVOL": "{:.1f}%",
            "ENTRY": "{:.2f}",
            "NOW": "{:.2f}",
            "TP": "{:.2f}",
            "SL": "{:.2f}",
            "PROFIT": "{:.1f}%",
            "%TO TP": "{:.1f}%",
            "RSI 5M": "{:.1f}",
            "SCORE": "{:.2f}",
        })
        .map(style_base, subset=["NO"])
        .map(style_emiten, subset=["EMITEN"])
        .map(style_gain, subset=["GAIN"])
        .map(style_wick, subset=["WICK"])
        .map(style_aksi, subset=["AKSI"])
        .map(style_sinyal, subset=["SINYAL"])
        .map(style_rvol, subset=["RVOL"])
        .map(style_price, subset=["ENTRY", "NOW"])
        .map(style_tp, subset=["TP"])
        .map(style_sl, subset=["SL"])
        .map(style_profit, subset=["PROFIT"])
        .map(style_to_tp, subset=["%TO TP"])
        .map(style_rsi_sig, subset=["RSI SIG"])
        .map(style_rsi, subset=["RSI 5M"])
        .map(style_val, subset=["VAL"])
        .map(style_fase, subset=["FASE"])
        .map(style_trend, subset=["TREND"])
        .map(style_score, subset=["SCORE"])
        .map(style_rank, subset=["RANK"])
        .map(style_prob, subset=["PROB"])
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#123b6d"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("border", "1px solid #1e293b"),
                ("font-size", "13px")
            ]},
            {"selector": "td", "props": [
                ("border", "1px solid #0f172a"),
                ("font-size", "12px"),
                ("padding", "4px 6px"),
                ("text-align", "center")
            ]}
        ])
    )

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Filter Screener")
    min_score = st.slider("Minimum Score Premium", 0, 100, 70)
    top_n = st.number_input("Jumlah baris tampil", min_value=20, max_value=2000, value=200, step=20)
    show_only_high = st.checkbox("Hanya HIGH Probability", value=False)
    bull_only = st.checkbox("Hanya Trend BULL", value=False)
    search_text = st.text_input("Cari Emiten", "")
    auto_refresh = st.checkbox("Auto Refresh 15 menit", value=False)
    run_btn = st.button("🔄 Scan Semua IDX", type="primary", use_container_width=True)

symbols = get_idx_symbols()

m1, m2, m3 = st.columns(3)
m1.metric("Total Symbol IDX", len(symbols))
m2.metric("Mode", "NO PRICE LIMIT")
m3.metric("Scan Basis", "Momentum Premium")

if run_btn or auto_refresh:
    with st.spinner("Sedang scan semua saham IDX... mohon tunggu..."):
        df = run_scan(symbols)

    if df.empty:
        st.error("Tidak ada data yang berhasil discan. Coba lagi.")
    else:
        filtered = df.copy()

        if min_score > 0:
            filtered = filtered[filtered["SCORE"] >= min_score]

        if show_only_high:
            filtered = filtered[filtered["PROB"] == "HIGH"]

        if bull_only:
            filtered = filtered[filtered["TREND"] == "BULL"]

        if search_text.strip():
            q = search_text.strip().upper()
            filtered = filtered[filtered["EMITEN"].str.contains(q, na=False)]

        filtered = filtered.head(int(top_n)).reset_index(drop=True)
        filtered["NO"] = range(1, len(filtered) + 1)

        high_count = int((filtered["PROB"] == "HIGH").sum())
        medium_count = int((filtered["PROB"] == "MEDIUM").sum())
        low_count = int((filtered["PROB"] == "LOW").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data Tampil", len(filtered))
        c2.metric("HIGH", high_count)
        c3.metric("MEDIUM", medium_count)
        c4.metric("LOW", low_count)

        st.dataframe(
            style_dataframe(filtered),
            use_container_width=True,
            height=760
        )

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="idx_momentum_premium_screener.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.subheader("Top 20 Premium Ranking")
        st.dataframe(filtered.head(20), use_container_width=True, height=500)
else:
    st.info("Klik **Scan Semua IDX** untuk mulai.")

st.markdown("---")
st.caption("Kolom SCORE adalah ranking score premium berbasis momentum, volume, RSI, trend, wick, timing entry, sinyal, fase, dan potensi profit.")
