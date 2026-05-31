import os
import pandas as pd
import datetime
import concurrent.futures
import yfinance as yf
import streamlit as st
import io

# ── Setup ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="5-Phase Monthly Screener",
    page_icon="📈",
    layout="wide"
)

# ── Custom Styling ───────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    h1 { color: #c084fc !important; }
    h2, h3 { color: #a855f7 !important; }
    .stButton>button {
        background-color: #7c3aed;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #6d28d9;
    }
    .stDownloadButton>button {
        background-color: #16a34a;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        border: none;
    }
    .metric-card {
        background: #1e1b4b;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #4c1d95;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INDICATOR MATH (No external TA library needed)
# ==========================================
def calc_rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=length - 1, adjust=False).mean()
    ema_down = down.ewm(com=length - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))


def calc_supertrend(high, low, close, period=7, multiplier=1):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    hl2 = (high + low) / 2
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)
    supertrend_dir = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > final_upperband.iloc[i - 1]:
            supertrend_dir.iloc[i] = 1
        elif close.iloc[i] < final_lowerband.iloc[i - 1]:
            supertrend_dir.iloc[i] = -1
        else:
            supertrend_dir.iloc[i] = supertrend_dir.iloc[i - 1]
            if supertrend_dir.iloc[i] == 1 and final_lowerband.iloc[i] < final_lowerband.iloc[i - 1]:
                final_lowerband.iloc[i] = final_lowerband.iloc[i - 1]
            if supertrend_dir.iloc[i] == -1 and final_upperband.iloc[i] > final_upperband.iloc[i - 1]:
                final_upperband.iloc[i] = final_upperband.iloc[i - 1]
    return supertrend_dir


# ==========================================
# SYMBOL LOADING
# ==========================================
def get_universe_from_csv(filepath):
    df = pd.read_csv(filepath)
    col_name = next((c for c in df.columns if c.strip().lower() == 'symbol'), df.columns[0])
    raw_symbols = df[col_name].dropna().astype(str).str.strip().str.upper().tolist()
    final_universe = []
    for s in raw_symbols:
        base = s.replace("NSE:", "").split("-")[0]
        final_universe.append(f"{base}.NS")
    return list(dict.fromkeys(final_universe))


# ==========================================
# FETCH & CALCULATE (5-PHASE LOGIC)
# ==========================================
def fetch_and_calculate(symbol):
    try:
        df = yf.download(symbol, period="10y", interval="1mo", progress=False)
        if df.empty:
            return {"Symbol": symbol.replace('.NS', ''), "Status": "No Data", "Close": 0.0, "Bench_Low": "-"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.dropna(subset=['Close'], inplace=True)
        if len(df) < 20:
            return {"Symbol": symbol.replace('.NS', ''), "Status": "Insufficient Data", "Close": float(df.iloc[-1]['Close']), "Bench_Low": "-"}

        df['RSI_14'] = calc_rsi(df['Close'], length=14)
        basis = df['Close'].rolling(window=20).mean()
        dev = 2 * df['Close'].rolling(window=20).std()
        df['Upper_BB'] = basis + dev
        df['ST_Dir'] = calc_supertrend(df['High'], df['Low'], df['Close'], period=7, multiplier=1)
        df['Prev_12M_High'] = df['High'].shift(1).rolling(window=12).max()
        df.dropna(inplace=True)

        if df.empty:
            return {"Symbol": symbol.replace('.NS', ''), "Status": "Insufficient Data", "Close": 0.0, "Bench_Low": "-"}

        state = "No Setup"
        benchmark_low = None
        for _, row in df.iterrows():
            if state in ["No Setup", "Phase 5: Bearish"]:
                if row['Close'] > row['Prev_12M_High'] and row['RSI_14'] >= 65 and row['Close'] > row['Upper_BB']:
                    state = "Phase 1: Monthly Breakout"
                    benchmark_low = None
            elif state in ["Phase 1: Monthly Breakout", "Phase 2: Continuation", "Phase 4: Secondary Breakout"]:
                if row['ST_Dir'] == 1:
                    state = "Phase 2: Continuation"
                else:
                    state = "Phase 3: Consolidation"
                    benchmark_low = row['Low']
            elif state == "Phase 3: Consolidation":
                if row['Close'] < benchmark_low:
                    state = "Phase 5: Bearish"
                elif row['ST_Dir'] == 1:
                    state = "Phase 4: Secondary Breakout"

        bench_low_val = f"{benchmark_low:.2f}" if benchmark_low is not None else "-"
        return {
            "Symbol": symbol.replace('.NS', ''),
            "Status": state,
            "Close": round(float(df.iloc[-1]['Close']), 2),
            "Bench_Low": bench_low_val
        }
    except Exception as e:
        return {"Symbol": symbol.replace('.NS', ''), "Status": f"Error: {e}", "Close": 0.0, "Bench_Low": "-"}


# ==========================================
# STREAMLIT UI
# ==========================================
st.title("☁️ 5-Phase Monthly Screener")
st.markdown("*Powered by Yahoo Finance • Runs 24/7 on the cloud*")
st.divider()

csv_path = os.path.join(BASE_DIR, "Symbols.csv")

if not os.path.exists(csv_path):
    st.error("❌ Symbols.csv not found! Please upload it to your GitHub repository.")
    st.stop()

symbols = get_universe_from_csv(csv_path)
st.info(f"📄 **{len(symbols)} stocks** loaded from Symbols.csv and ready to scan.")

st.divider()

if st.button("🚀 Start Screener"):
    watchlist_phases = ["Phase 2: Continuation", "Phase 3: Consolidation", "Phase 4: Secondary Breakout"]
    results = []

    progress_bar = st.progress(0, text="Connecting to Yahoo Finance...")
    log_area = st.empty()
    log_lines = []

    total = len(symbols)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_and_calculate, s): s for s in symbols}
        completed = 0
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            results.append(res)
            completed += 1
            pct = completed / total
            log_lines.append(f"✔️ {res['Symbol']} → {res['Status']}")
            if len(log_lines) > 12:
                log_lines = log_lines[-12:]
            progress_bar.progress(pct, text=f"Processing {completed}/{total} stocks...")
            log_area.code("\n".join(log_lines), language=None)

    progress_bar.progress(1.0, text="✅ Scan complete!")
    log_area.empty()

    df_results = pd.DataFrame(results)

    # ── Summary Stats ───────────────────────
    st.divider()
    st.subheader("📊 Scan Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Scanned", total)
    c2.metric("Phase 1: Breakout", len(df_results[df_results['Status'] == 'Phase 1: Monthly Breakout']))
    c3.metric("Phase 2: Continuation", len(df_results[df_results['Status'] == 'Phase 2: Continuation']))
    c4.metric("Phase 3: Consolidation", len(df_results[df_results['Status'] == 'Phase 3: Consolidation']))
    c5.metric("Phase 4: Sec. Breakout", len(df_results[df_results['Status'] == 'Phase 4: Secondary Breakout']))

    # ── Watchlist Table ─────────────────────
    st.divider()
    st.subheader("⭐ Watchlist Stocks (Phases 2, 3 & 4)")
    df_watchlist = df_results[df_results['Status'].isin(watchlist_phases)].reset_index(drop=True)
    if df_watchlist.empty:
        st.warning("No stocks matched the watchlist criteria today.")
    else:
        st.dataframe(df_watchlist, use_container_width=True, height=400)

    # ── Download Buttons ────────────────────
    st.divider()
    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)

    # Excel download
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='All Results')
        df_watchlist.to_excel(writer, index=False, sheet_name='Watchlist')
    excel_buffer.seek(0)

    col1.download_button(
        label="📥 Download Full Excel (All Stocks)",
        data=excel_buffer,
        file_name=f"Screener2_YF_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # CSV download
    csv_data = df_watchlist[['Symbol']].to_csv(index=False).encode('utf-8')
    col2.download_button(
        label="📥 Download M_screen.csv (Watchlist Only)",
        data=csv_data,
        file_name="M_screen.csv",
        mime="text/csv"
    )

    # ── Full Results Table ──────────────────
    st.divider()
    st.subheader("📋 Full Results (All Stocks)")
    st.dataframe(df_results, use_container_width=True, height=500)
