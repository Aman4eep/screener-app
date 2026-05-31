import os
import time
import pandas as pd
import datetime
import sys
import traceback
import concurrent.futures
import flet as ft
import yfinance as yf
import threading

# ── Setup path ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

# ==========================================
# INDICATOR MATH (No pandas-ta or numba needed)
# ==========================================
def calc_rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=length-1, adjust=False).mean()
    ema_down = down.ewm(com=length-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calc_supertrend(high, low, close, period=7, multiplier=1):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    hl2 = (high + low) / 2
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)
    
    supertrend_dir = pd.Series(1, index=close.index) 
    
    for i in range(1, len(close)):
        if close.iloc[i] > final_upperband.iloc[i-1]:
            supertrend_dir.iloc[i] = 1
        elif close.iloc[i] < final_lowerband.iloc[i-1]:
            supertrend_dir.iloc[i] = -1
        else:
            supertrend_dir.iloc[i] = supertrend_dir.iloc[i-1]
            if supertrend_dir.iloc[i] == 1 and final_lowerband.iloc[i] < final_lowerband.iloc[i-1]:
                final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            if supertrend_dir.iloc[i] == -1 and final_upperband.iloc[i] > final_upperband.iloc[i-1]:
                final_upperband.iloc[i] = final_upperband.iloc[i-1]
                
    return supertrend_dir

# ==========================================
# 1. SYMBOL MASTER LIST MATCHING (YAHOO FINANCE)
# ==========================================
def get_universe_from_csv(filepath, log_callback):
    try:
        df = pd.read_csv(filepath)
        col_name = next((c for c in df.columns if c.strip().lower() == 'symbol'), df.columns[0])
        raw_symbols = df[col_name].dropna().astype(str).str.strip().str.upper().tolist()
        log_callback(f"📄 Loaded {len(raw_symbols)} base symbols from {os.path.basename(filepath)}")

        final_universe = []
        for s in raw_symbols:
            base = s.replace("NSE:", "").split("-")[0]
            yf_symbol = f"{base}.NS"
            final_universe.append(yf_symbol)
            
        final_universe = list(dict.fromkeys(final_universe))
        log_callback(f"✅ Formatted {len(final_universe)} symbols for Yahoo Finance.")
        return final_universe
    except Exception as e:
        log_callback(f"❌ Failed to process symbols: {e}")
        raise e

# ==========================================
# 2. FETCH & CALCULATE (5-PHASE LOGIC)
# ==========================================
def fetch_and_calculate(symbol, log_callback):
    try:
        df = yf.download(symbol, period="10y", interval="1mo", progress=False)
        if df.empty:
            return {"Symbol": symbol, "Status": "No Data", "Close": 0.0, "Bench_Low": "-"}
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
            
        df.dropna(subset=['Close'], inplace=True)
        
        if len(df) < 20:
            return {"Symbol": symbol, "Status": "Insufficient Data", "Close": float(df.iloc[-1]['Close']), "Bench_Low": "-"}

        # Use our custom Math indicators instead of pandas-ta
        df['RSI_14'] = calc_rsi(df['Close'], length=14)
        
        # Bollinger Bands Upper
        basis = df['Close'].rolling(window=20).mean()
        dev = 2 * df['Close'].rolling(window=20).std()
        df['Upper_BB'] = basis + dev

        # Supertrend
        df['ST_Dir'] = calc_supertrend(df['High'], df['Low'], df['Close'], period=7, multiplier=1)

        df['Prev_12M_High'] = df['High'].shift(1).rolling(window=12).max()
        df.dropna(inplace=True)

        if df.empty:
            return {"Symbol": symbol, "Status": "Insufficient Data", "Close": 0.0, "Bench_Low": "-"}

        state         = "No Setup"
        benchmark_low = None

        for _, row in df.iterrows():
            if state in ["No Setup", "Phase 5: Bearish"]:
                if (row['Close'] > row['Prev_12M_High'] and row['RSI_14'] >= 65 and row['Close'] > row['Upper_BB']):
                    state         = "Phase 1: Monthly Breakout"
                    benchmark_low = None
            elif state in ["Phase 1: Monthly Breakout", "Phase 2: Continuation", "Phase 4: Secondary Breakout"]:
                if row['ST_Dir'] == 1:
                    state = "Phase 2: Continuation"
                else:
                    state         = "Phase 3: Consolidation"
                    benchmark_low = row['Low']
            elif state == "Phase 3: Consolidation":
                if row['Close'] < benchmark_low:
                    state = "Phase 5: Bearish"
                elif row['ST_Dir'] == 1:
                    state = "Phase 4: Secondary Breakout"

        bench_low_val = f"{benchmark_low:.2f}" if benchmark_low is not None else "-"
        return {
            "Symbol":    symbol.replace('.NS', ''),
            "Status":    state,
            "Close":     round(float(df.iloc[-1]['Close']), 2),
            "Bench_Low": bench_low_val
        }
    except Exception as e:
        return {"Symbol": symbol.replace('.NS', ''), "Status": f"Error: {e}", "Close": 0.0, "Bench_Low": "-"}

# ==========================================
# 3. FLET USER INTERFACE
# ==========================================
def main(page: ft.Page):
    page.title = "5-Phase Monthly Screener (Cloud Edition)"
    page.theme_mode = ft.ThemeMode.DARK
    page.scroll = ft.ScrollMode.ADAPTIVE
    page.padding = 20

    title = ft.Text("☁️ Cloud Monthly Screener", size=26, weight=ft.FontWeight.BOLD, color=ft.Colors.PURPLE_300)
    subtitle = ft.Text("Runs 24/7. Results can be downloaded to your phone.", size=14, color=ft.Colors.GREY_400)
    
    start_btn = ft.ElevatedButton("Start Screener", icon=ft.Icons.PLAY_ARROW, bgcolor=ft.Colors.PURPLE_700, color=ft.Colors.WHITE, height=50)
    progress_ring = ft.ProgressRing(visible=False, width=20, height=20, color=ft.Colors.PURPLE_300)
    progress_status = ft.Text("", size=14, color=ft.Colors.GREY_300)
    
    download_btn = ft.ElevatedButton("📥 Download Excel Results", bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE, height=50, visible=False)
    
    log_box = ft.ListView(expand=True, spacing=5, auto_scroll=True, height=180)
    log_container = ft.Container(
        content=log_box,
        border=ft.Border.all(1, ft.Colors.GREY_700),
        border_radius=10,
        padding=10,
        bgcolor=ft.Colors.BLACK,
        visible=False
    )
    
    results_title = ft.Text("Screener Results", size=20, weight=ft.FontWeight.BOLD, visible=False)
    results_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Symbol")),
            ft.DataColumn(ft.Text("Status")),
            ft.DataColumn(ft.Text("Close Price"), numeric=True),
            ft.DataColumn(ft.Text("Bench Low"), numeric=True),
        ],
        rows=[],
        visible=False
    )
    
    results_container = ft.Column(
        controls=[
            ft.Row([results_title, download_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN), 
            results_table
        ],
        visible=False,
        scroll=ft.ScrollMode.ADAPTIVE
    )

    def write_log(message):
        print(message)
        log_box.controls.append(ft.Text(message, size=12, color=ft.Colors.GREEN_300, font_family="monospace"))
        page.update()

    def _background_screener():
        try:
            write_log("🌐 Connecting to Yahoo Finance...")

            csv_path = os.path.join(BASE_DIR, "Symbols.csv")
            if not os.path.exists(csv_path):
                write_log(f"❌ Symbols.csv not found at: {csv_path}")
                progress_status.value = "Error: Symbols.csv missing"
                page.update()
                return

            symbols = get_universe_from_csv(csv_path, write_log)
            if not symbols:
                write_log("⚠️ No symbols loaded from CSV.")
                progress_status.value = "No symbols found."
                page.update()
                return

            results = []
            total_symbols = len(symbols)
            progress_status.value = f"Processing 0/{total_symbols} stocks..."
            page.update()

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(fetch_and_calculate, s, write_log): s for s in symbols}
                completed_count = 0
                for f in concurrent.futures.as_completed(futures):
                    res = f.result()
                    results.append(res)
                    completed_count += 1
                    progress_status.value = f"Processing {completed_count}/{total_symbols} stocks..."
                    write_log(f"✔️ {res['Symbol']} → {res['Status']}")

            write_log("💾 Saving screener results...")
            
            excel_filename = f"Screener2_YF_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
            output_excel = os.path.join(ASSETS_DIR, excel_filename)
            
            df_results = pd.DataFrame(results)
            df_results.to_excel(output_excel, index=False)
            write_log(f"💾 Full Excel ready for download!")

            watchlist_phases = ["Phase 2: Continuation", "Phase 3: Consolidation", "Phase 4: Secondary Breakout"]
            df_watchlist = df_results[df_results["Status"].isin(watchlist_phases)][["Symbol"]]

            m_screen_path = os.path.join(ASSETS_DIR, "M_screen.csv")
            df_watchlist.to_csv(m_screen_path, index=False)
            
            download_btn.on_click = lambda e: page.launch_url(f"/{excel_filename}")
            download_btn.visible = True

            for r in results:
                status_color = ft.Colors.WHITE
                if r['Status'] in watchlist_phases:
                    status_color = ft.Colors.GREEN_400
                elif "Error" in r['Status'] or "No Data" in r['Status']:
                    status_color = ft.Colors.RED_300
                
                results_table.rows.append(
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(r['Symbol'], weight=ft.FontWeight.BOLD)),
                        ft.DataCell(ft.Text(r['Status'], color=status_color)),
                        ft.DataCell(ft.Text(str(r['Close']))),
                        ft.DataCell(ft.Text(str(r['Bench_Low']))),
                    ])
                )

            results_title.visible = True
            results_table.visible = True
            results_container.visible = True
            progress_status.value = f"Done! Screener completed successfully."
            write_log("🎉 Screener Finished!")

        except Exception as ex:
            write_log(f"❌ Error encountered: {ex}")
            progress_status.value = f"Failed with error: {ex}"
            traceback.print_exc()

        finally:
            start_btn.disabled = False
            progress_ring.visible = False
            page.update()

    def run_screener_task(e):
        start_btn.disabled = True
        download_btn.visible = False
        progress_ring.visible = True
        log_box.controls.clear()
        results_container.visible = False
        results_table.rows.clear()
        log_container.visible = True
        progress_status.value = "Starting screener process..."
        page.update()
        threading.Thread(target=_background_screener, daemon=True).start()

    start_btn.on_click = run_screener_task

    page.add(
        ft.Container(
            content=ft.Column(
                controls=[
                    title,
                    subtitle,
                    ft.Divider(height=10, color=ft.Colors.GREY_700),
                    ft.Row(
                        controls=[start_btn, progress_ring, progress_status],
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER
                    ),
                    ft.Divider(height=10, color="transparent"),
                    log_container,
                    ft.Divider(height=15, color="transparent"),
                    results_container
                ]
            ),
            padding=10
        )
    )

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    port = int(os.environ.get("PORT", 8550))
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=port, host="0.0.0.0", assets_dir="assets")
