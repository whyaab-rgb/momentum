# app.py
# Streamlit Screener Saham Akumulasi - RVOL Tinggi
# Jalankan:
# pip install streamlit yfinance pandas numpy
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Screener Akumulasi RVOL", layout="wide")

# =========================
# CONFIG
# =========================
DEFAULT_SYMBOLS = [
    "BBRI","BMRI","BBCA","TLKM","ASII","BRIS","ANTM","PGAS","EXCL","ERAA",
    "GOTO","MEDC","PTBA","ADRO","INCO","TINS","BUKA","SIDO","MYOR","CPIN",
    "KLBF","JPFA","MAPI","PWON","SMGR","AKRA","ESSA","SCMA","TOWR","MDKA",
    "ICBP","INDF","UNVR","HMSP","ITMG","HRUM","PTPP","WIKA","WSKT","ADHI",
    "LSIP","AALI","SSMS","TBLA","BISI","MAIN","BRPT","FREN","ISAT","LINK",
    "MNCN","VIVA","LPPF","RALS","ACES","MAPA","SRIL","PBRX","ARGO","RICY",
    "AUTO","SMSM","GJTL","IMAS","NIPS","PRAS","TRIS","UNTR","HEAL","SILO",
    "MIKA","CARE","SAME","KAEF","INAF","PYFA","TSPC","SOHO","PEVE","SRAJ",
    "BANK","BBYB","BBHI","BBKP","BBMD","BBNI","BBTN","BDMN","MEGA","PNBN",
    "BNGA","BNII","BACA","SDRA","MAYA","AGRO","MCOR","DNAR","NOBU","ARTO",
    "BFIN","ADMF","IMFI","WOMF","CFIN","HDFA","VRNA","TIFA","TRUS","DEFI",
    "SRTG","ABBA","BHIT","MFIN","APIC","LPKR","BSDE","CTRA","SMRA","PWON",
    "DILD","KIJA","DMAS","BEST","GPRA","LPCK","NRCA","TOTL","WEGE","WTON",
    "INTP","SMBR","CSAP","RANC","HERO","AMRT","MIDI","DAYA","MAPB","MPPA",
    "ULTJ","CLEO","DLTA","ICBP","ROTI","SKLT","STTP","AISA","CEKA","GOOD",
]
]

# =========================
# HELPERS
# =========================
def safe_div(a, b):
    try:
        if b == 0 or pd.isna(b):
            return np.nan
        return a / b
    except:
        return np.nan

def fmt_num(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"

def fmt_int(x):
    if pd.isna(x):
        return "-"
    return f"{int(x):,}"

def human_value(x):
    if pd.isna(x):
        return "-"
    x = float(x)
    if x >= 1_000_000_000_000:
        return f"{x/1_000_000_000_000:.2f} T"
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f} B"
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f} M"
    return f"{x:,.0f}"

def normalize_symbols(text):
    raw = [x.strip().upper() for x in text.replace("\n", ",").split(",") if x.strip()]
    out = []
    for s in raw:
        if not s.endswith(".JK"):
            s = s + ".JK"
        out.append(s)
    return list(dict.fromkeys(out))

def download_one(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )
        if df is None or df.empty:
            return None

        # Handle possible multiindex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        needed = ["Open", "High", "Low", "Close", "Volume"]
        for c in needed:
            if c not in df.columns:
                return None

        df = df[needed].copy()
        df.dropna(inplace=True)

        if len(df) < 40:
            return None

        return df
    except:
        return None

def compute_features(df):
    d = df.copy()

    d["Value"] = d["Close"] * d["Volume"]
    d["AvgVol20"] = d["Volume"].rolling(20).mean()
    d["AvgVal20"] = d["Value"].rolling(20).mean()
    d["RVOL"] = d["Volume"] / d["AvgVol20"]

    d["EMA10"] = d["Close"].ewm(span=10, adjust=False).mean()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()

    d["HH20"] = d["High"].rolling(20).max()
    d["LL20"] = d["Low"].rolling(20).min()

    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - d["Close"].shift(1)).abs()
    tr3 = (d["Low"] - d["Close"].shift(1)).abs()
    d["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["ATR14"] = d["TR"].rolling(14).mean()

    # Money Flow sederhana
    mf_mult = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / (d["High"] - d["Low"]).replace(0, np.nan)
    mf_mult = mf_mult.replace([np.inf, -np.inf], np.nan).fillna(0)
    d["MFV"] = mf_mult * d["Volume"]
    d["ADL"] = d["MFV"].cumsum()
    d["ADL_Slope5"] = d["ADL"].diff(5)

    # OBV
    direction = np.sign(d["Close"].diff()).fillna(0)
    d["OBV"] = (direction * d["Volume"]).cumsum()
    d["OBV_Slope5"] = d["OBV"].diff(5)

    # Posisi close dalam range harian
    d["ClosePos"] = (d["Close"] - d["Low"]) / (d["High"] - d["Low"]).replace(0, np.nan)

    # Distance to breakout
    d["PrevHH20"] = d["High"].shift(1).rolling(20).max()
    d["BreakoutPct"] = ((d["Close"] / d["PrevHH20"]) - 1) * 100

    return d

def score_accumulation(d):
    last = d.iloc[-1]
    prev = d.iloc[-2]

    score = 0
    notes = []

    # 1) RVOL
    rvol = last["RVOL"]
    if pd.notna(rvol):
        if rvol >= 3.0:
            score += 30
            notes.append("RVOL sangat tinggi")
        elif rvol >= 2.0:
            score += 22
            notes.append("RVOL tinggi")
        elif rvol >= 1.5:
            score += 14
            notes.append("RVOL di atas normal")

    # 2) Nilai transaksi
    val = last["Value"]
    if pd.notna(val):
        if val >= 50_000_000_000:
            score += 15
            notes.append("nilai transaksi besar")
        elif val >= 20_000_000_000:
            score += 10
            notes.append("nilai transaksi cukup")
        elif val >= 10_000_000_000:
            score += 6
            notes.append("nilai transaksi lumayan")

    # 3) Struktur harga
    if last["Close"] > last["EMA20"]:
        score += 8
        notes.append("close di atas EMA20")
    if last["EMA20"] > last["EMA50"]:
        score += 8
        notes.append("EMA20 di atas EMA50")
    if last["Close"] > prev["Close"]:
        score += 6
        notes.append("harga menguat")
    if pd.notna(last["ClosePos"]) and last["ClosePos"] >= 0.7:
        score += 8
        notes.append("close dekat high")

    # 4) Akumulasi flow
    if pd.notna(last["ADL_Slope5"]) and last["ADL_Slope5"] > 0:
        score += 8
        notes.append("arus akumulasi positif")
    if pd.notna(last["OBV_Slope5"]) and last["OBV_Slope5"] > 0:
        score += 7
        notes.append("OBV naik")

    # 5) Dekat breakout
    if pd.notna(last["BreakoutPct"]):
        if -2.0 <= last["BreakoutPct"] <= 2.0:
            score += 10
            notes.append("dekat area breakout")
        elif last["BreakoutPct"] > 2.0:
            score += 6
            notes.append("sudah lepas breakout")

    # Penalti
    if last["Close"] < last["EMA50"]:
        score -= 10
    if pd.notna(rvol) and rvol < 1:
        score -= 10

    # Label
    if score >= 75:
        grade = "A"
    elif score >= 60:
        grade = "B"
    elif score >= 45:
        grade = "C"
    else:
        grade = "D"

    return score, grade, ", ".join(notes[:5])

def build_trade_plan(d):
    last = d.iloc[-1]
    prev_hh20 = last["PrevHH20"]
    atr = last["ATR14"]
    close = last["Close"]
    low = last["Low"]
    ema20 = last["EMA20"]

    buy_zone_low = min(close, prev_hh20) if pd.notna(prev_hh20) else close
    buy_zone_high = max(close, prev_hh20) if pd.notna(prev_hh20) else close

    if pd.notna(atr):
        stop_loss = min(low, ema20) - (0.3 * atr)
        take_profit_1 = close + (1.0 * atr)
        take_profit_2 = close + (2.0 * atr)
    else:
        stop_loss = min(low, ema20) * 0.97
        take_profit_1 = close * 1.05
        take_profit_2 = close * 1.10

    risk_pct = ((close - stop_loss) / close) * 100 if close and stop_loss else np.nan
    upside_pct = ((take_profit_2 - close) / close) * 100 if close and take_profit_2 else np.nan

    return {
        "Entry Zone": f"{buy_zone_low:.0f} - {buy_zone_high:.0f}",
        "Stop Loss": round(stop_loss, 2) if pd.notna(stop_loss) else np.nan,
        "TP1": round(take_profit_1, 2) if pd.notna(take_profit_1) else np.nan,
        "TP2": round(take_profit_2, 2) if pd.notna(take_profit_2) else np.nan,
        "Risk %": round(risk_pct, 2) if pd.notna(risk_pct) else np.nan,
        "Upside %": round(upside_pct, 2) if pd.notna(upside_pct) else np.nan,
    }

def analyze_symbol(symbol):
    df = download_one(symbol)
    if df is None:
        return None

    d = compute_features(df)
    if d is None or len(d) < 30:
        return None

    last = d.iloc[-1]
    prev = d.iloc[-2]
    score, grade, notes = score_accumulation(d)
    plan = build_trade_plan(d)

    price = float(last["Close"])
    chg_pct = ((last["Close"] / prev["Close"]) - 1) * 100 if prev["Close"] else np.nan
    breakout_level = float(last["PrevHH20"]) if pd.notna(last["PrevHH20"]) else np.nan
    support = float(d["Low"].tail(10).min())
    resistance = float(d["High"].tail(10).max())

    return {
        "Ticker": symbol.replace(".JK", ""),
        "Price": round(price, 2),
        "Change %": round(chg_pct, 2) if pd.notna(chg_pct) else np.nan,
        "Volume": float(last["Volume"]),
        "Avg Vol 20": float(last["AvgVol20"]) if pd.notna(last["AvgVol20"]) else np.nan,
        "RVOL": round(float(last["RVOL"]), 2) if pd.notna(last["RVOL"]) else np.nan,
        "Value": float(last["Value"]) if pd.notna(last["Value"]) else np.nan,
        "EMA20": round(float(last["EMA20"]), 2) if pd.notna(last["EMA20"]) else np.nan,
        "EMA50": round(float(last["EMA50"]), 2) if pd.notna(last["EMA50"]) else np.nan,
        "ATR14": round(float(last["ATR14"]), 2) if pd.notna(last["ATR14"]) else np.nan,
        "Breakout Lv": round(breakout_level, 2) if pd.notna(breakout_level) else np.nan,
        "Support": round(support, 2) if pd.notna(support) else np.nan,
        "Resistance": round(resistance, 2) if pd.notna(resistance) else np.nan,
        "Entry Zone": plan["Entry Zone"],
        "Stop Loss": plan["Stop Loss"],
        "TP1": plan["TP1"],
        "TP2": plan["TP2"],
        "Risk %": plan["Risk %"],
        "Upside %": plan["Upside %"],
        "Score": score,
        "Grade": grade,
        "Signal": "AKUMULASI" if score >= 60 else ("PANTAU" if score >= 45 else "LEWATI"),
        "Notes": notes
    }

def style_dataframe(df):
    def color_signal(v):
        if v == "AKUMULASI":
            return "background-color: #123524; color: #7CFC98; font-weight: bold;"
        elif v == "PANTAU":
            return "background-color: #3b2f0b; color: #ffd966; font-weight: bold;"
        return "background-color: #3a0f10; color: #ff8080;"

    def color_grade(v):
        if v == "A":
            return "background-color: #123524; color: #7CFC98; font-weight: bold;"
        elif v == "B":
            return "background-color: #1a2f4f; color: #9fd3ff; font-weight: bold;"
        elif v == "C":
            return "background-color: #3b2f0b; color: #ffd966; font-weight: bold;"
        return "background-color: #3a0f10; color: #ff8080;"

    def color_rvol(v):
        if pd.isna(v):
            return ""
        if v >= 3:
            return "background-color: #0b3d2e; color: #7CFC98; font-weight: bold;"
        elif v >= 2:
            return "background-color: #173f5f; color: #9fd3ff; font-weight: bold;"
        elif v >= 1.5:
            return "background-color: #4b3b12; color: #ffd966;"
        return ""

    def color_score(v):
        if pd.isna(v):
            return ""
        if v >= 75:
            return "background-color: #123524; color: #7CFC98; font-weight: bold;"
        elif v >= 60:
            return "background-color: #173f5f; color: #9fd3ff; font-weight: bold;"
        elif v >= 45:
            return "background-color: #4b3b12; color: #ffd966;"
        return "background-color: #3a0f10; color: #ff8080;"

    def color_change(v):
        if pd.isna(v):
            return ""
        if v > 0:
            return "color: #7CFC98; font-weight: bold;"
        elif v < 0:
            return "color: #ff8080; font-weight: bold;"
        return ""

    styler = (
        df.style
        .map(color_signal, subset=["Signal"])
        .map(color_grade, subset=["Grade"])
        .map(color_rvol, subset=["RVOL"])
        .map(color_score, subset=["Score"])
        .map(color_change, subset=["Change %"])
        .format({
            "Price": "{:,.2f}",
            "Change %": "{:,.2f}",
            "Volume": lambda x: human_value(x),
            "Avg Vol 20": lambda x: human_value(x),
            "RVOL": "{:,.2f}",
            "Value": lambda x: human_value(x),
            "EMA20": "{:,.2f}",
            "EMA50": "{:,.2f}",
            "ATR14": "{:,.2f}",
            "Breakout Lv": "{:,.2f}",
            "Support": "{:,.2f}",
            "Resistance": "{:,.2f}",
            "Stop Loss": "{:,.2f}",
            "TP1": "{:,.2f}",
            "TP2": "{:,.2f}",
            "Risk %": "{:,.2f}",
            "Upside %": "{:,.2f}",
            "Score": "{:,.0f}"
        })
    )
    return styler

# =========================
# UI
# =========================
st.title("📈 Screener Saham Akumulasi - RVOL Tinggi")
st.caption("Mencari saham dengan indikasi akumulasi: RVOL tinggi, harga kuat, arus volume positif, dan area masuk yang lebih jelas.")

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    symbols_text = st.text_area(
        "Daftar saham (pisahkan dengan koma, otomatis tambah .JK)",
        value=",".join([s.replace(".JK", "") for s in DEFAULT_SYMBOLS]),
        height=140
    )

with col2:
    min_rvol = st.number_input("Min RVOL", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
    min_score = st.number_input("Min Score", min_value=0, max_value=100, value=45, step=1)

with col3:
    min_value_b = st.number_input("Min Nilai Transaksi (B)", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
    only_above_ema20 = st.checkbox("Hanya > EMA20", value=True)

with col4:
    top_n = st.number_input("Top N", min_value=5, max_value=200, value=25, step=5)
    auto_sort = st.selectbox("Urutkan", ["Score", "RVOL", "Value", "Change %"], index=0)

run_scan = st.button("🔍 Scan Saham", use_container_width=True)

if run_scan:
    symbols = normalize_symbols(symbols_text)
    if not symbols:
        st.warning("Masukkan minimal 1 saham.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    rows = []

    for i, sym in enumerate(symbols):
        status.info(f"Scanning {sym} ... ({i+1}/{len(symbols)})")
        result = analyze_symbol(sym)
        if result:
            rows.append(result)
        progress.progress((i + 1) / len(symbols))

    status.empty()
    progress.empty()

    if not rows:
        st.error("Tidak ada data yang berhasil diambil.")
        st.stop()

    df = pd.DataFrame(rows)

    # Filter
    df = df[df["RVOL"] >= min_rvol]
    df = df[df["Score"] >= min_score]
    df = df[df["Value"] >= (min_value_b * 1_000_000_000)]

    if only_above_ema20:
        df = df[df["Price"] > df["EMA20"]]

    if df.empty:
        st.warning("Tidak ada saham yang lolos filter.")
        st.stop()

    df = df.sort_values(by=auto_sort, ascending=False).head(int(top_n)).reset_index(drop=True)
    df.index = df.index + 1

    # Ringkasan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Lolos", len(df))
    c2.metric("Rata-rata RVOL", f"{df['RVOL'].mean():.2f}")
    c3.metric("Rata-rata Score", f"{df['Score'].mean():.1f}")
    c4.metric("Waktu Scan", datetime.now().strftime("%H:%M:%S"))

    st.subheader("Tabel Screener Akumulasi")
    show_cols = [
        "Ticker", "Signal", "Grade", "Score", "Price", "Change %",
        "RVOL", "Value", "Volume", "Avg Vol 20",
        "EMA20", "EMA50", "Breakout Lv", "Support", "Resistance",
        "Entry Zone", "Stop Loss", "TP1", "TP2", "Risk %", "Upside %",
        "Notes"
    ]
    st.dataframe(
        style_dataframe(df[show_cols]),
        use_container_width=True,
        height=700
    )

    st.subheader("Top Kandidat Akumulasi")
    top_df = df.head(5)[["Ticker", "Signal", "Grade", "Score", "RVOL", "Value", "Entry Zone", "Stop Loss", "TP1", "TP2", "Notes"]]
    st.dataframe(style_dataframe(top_df), use_container_width=True, height=260)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download hasil CSV",
        data=csv,
        file_name="screener_akumulasi_rvol.csv",
        mime="text/csv",
        use_container_width=True
    )

with st.expander("Cara baca tabel"):
    st.markdown("""
**Makna kolom penting:**

- **RVOL**: volume hari ini dibanding rata-rata 20 hari. Semakin tinggi, semakin menarik.
- **Value**: nilai transaksi. Membantu lihat apakah volume benar-benar bermakna.
- **Score**: skor gabungan untuk menilai potensi akumulasi.
- **Signal**:
  - **AKUMULASI** = kandidat paling menarik
  - **PANTAU** = menarik, tapi belum kuat
  - **LEWATI** = belum layak fokus
- **Breakout Lv**: area high 20 hari sebelumnya.
- **Entry Zone**: area masuk yang dipertimbangkan.
- **Stop Loss**: area invalidasi.
- **TP1 / TP2**: target awal dan lanjutan.

**Logika sederhananya:**
saham dianggap menarik bila volume melonjak, harga tidak lemah, dan arus volume menunjukkan akumulasi.
""")
