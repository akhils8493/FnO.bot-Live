import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import requests
import os
import pyotp
import time

# ---------------------------
# CONFIGURATION
# ---------------------------
API_KEY = "WM6i3ikL"
CLIENT_ID = "AABY364105"
PIN = "6954"
TOTP_TOKEN = "D5PMGU3B674K4YFIQNE7CKDUSU"
NIFTY_TOKEN = "99926000"  # Angel One NSE Nifty Spot Token

# STRATEGY SETTINGS
MIN_OPT_PRICE = 150   # Minimum Option Price to Enter
MAX_OPT_PRICE = 190   # Maximum Option Price to Enter
REFRESH_RATE_SEC = 5  # Check for data every 5 seconds

st.set_page_config(page_title="Nifty Strategy (Live)", layout="wide")

# ---------------------------
# ANGEL LOGIN
# ---------------------------
@st.cache_resource(ttl=600)
def angel_login():
    try:
        obj = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_TOKEN).now()
        data = obj.generateSession(CLIENT_ID, PIN, totp)
        if not data or not data.get("status"):
            st.error(f"Login failed: {data.get('message') if data else 'No response'}")
            return None
        return obj
    except Exception as e:
        st.error(f"Angel login error: {e}")
        return None

# ---------------------------
# UTILITIES
# ---------------------------
def get_next_tuesday(date=None):
    if date is None:
        date = datetime.today().date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    days_ahead = (1 - date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_tues = date + timedelta(days=days_ahead)
    return next_tues.strftime("%d%b%Y").upper()

def get_atm_strike(price):
    return round(price / 50) * 50

def check_and_fix_weekend(key):
    selected_date = st.session_state[key]
    if selected_date.weekday() == 5: # Saturday
        new_date = selected_date + timedelta(days=2)
        st.session_state[key] = new_date
        st.warning(f"Market is closed on Saturday. Date set to Monday, {new_date.strftime('%Y-%m-%d')}.")
    elif selected_date.weekday() == 6: # Sunday
        new_date = selected_date + timedelta(days=1)
        st.session_state[key] = new_date
        st.warning(f"Market is closed on Sunday. Date set to Monday, {new_date.strftime('%Y-%m-%d')}.")

# ---------------------------
# SCRIP MASTER
# ---------------------------
@st.cache_data(ttl=3600)
def get_scrip_master_df():
    json_file = "OpenAPIScripMaster.json"
    if not os.path.exists(json_file) or os.path.getsize(json_file) == 0:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(json_file, "wb") as f:
                f.write(r.content)
        except Exception:
            return pd.DataFrame()
    return pd.read_json(json_file)

def get_token_from_master_cached(df_master, symbol, expiry_date, strike, option_type):
    try:
        strike_val = float(strike) * 100
        suffix = option_type.upper()
        filtered = df_master[
            (df_master["exch_seg"] == "NFO") &
            (df_master["name"] == symbol) &
            (df_master["expiry"] == expiry_date) &
            (df_master["symbol"].str.endswith(suffix)) &
            (pd.to_numeric(df_master["strike"], errors="coerce") == strike_val)
        ]
        if not filtered.empty:
            return filtered.iloc[0]["token"]
        return None
    except Exception:
        return None

# ---------------------------
# FETCH DATA
# ---------------------------
def fetch_candle_data(api_obj, token, interval, exchange="NSE", specific_date=None):
    try:
        if specific_date:
            is_today = specific_date == datetime.today().date()
            from_dt = datetime.combine(specific_date, datetime.min.time()) + timedelta(hours=9, minutes=15)
            
            if is_today:
                to_dt = datetime.now()
            else:
                to_dt = datetime.combine(specific_date, datetime.min.time()) + timedelta(hours=15, minutes=30)

        params = {
            "exchange": exchange,
            "symboltoken": str(token),
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": to_dt.strftime("%Y-%m-%d %H:%M")
        }
        
        time.sleep(0.2) 
        resp = api_obj.getCandleData(params)
        
        if resp and resp.get("status") and resp.get("data"):
            data = resp.get("data", [])
            cols = ["datetime", "open", "high", "low", "close", "volume"]
            df = pd.DataFrame(data, columns=cols)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ---------------------------
# INDICATORS & STRATEGY
# ---------------------------
def add_custom_ema(df):
    df = df.copy()
    df["ema_3"] = df["close"].ewm(span=3, adjust=False).mean()
    df["ema_3_offset"] = df["ema_3"].shift(0) 
    df["ema_3_smooth"] = df["ema_3_offset"].ewm(span=9, adjust=False).mean()

    df["above_ema_alert"] = df["low"] > df["ema_3_smooth"] 
    df["below_ema_alert"] = df["high"] < df["ema_3_smooth"] 
    df["alert_candle"] = df["above_ema_alert"] | df["below_ema_alert"]
    return df

def validate_signal_with_options(api_obj, master_df, expiry_date, signal_time, signal_type, spot_price, trade_date):
    atm = get_atm_strike(spot_price)
    strikes = [atm + (i * 50) for i in range(-5, 6)]
    candidates = []
    
    for strike in strikes:
        token = get_token_from_master_cached(master_df, "NIFTY", expiry_date, strike, signal_type)
        if not token: continue
            
        df_opt = fetch_candle_data(api_obj, token, "TEN_MINUTE", exchange="NFO", specific_date=trade_date)
        if df_opt.empty: continue
            
        df_opt = add_custom_ema(df_opt)
        match_rows = df_opt[df_opt["datetime"] == signal_time]
        if match_rows.empty: continue
            
        idx = match_rows.index[0]
        row = df_opt.iloc[idx]
        
        entry_price = row["high"]
        if not (MIN_OPT_PRICE <= entry_price <= MAX_OPT_PRICE): continue
        if not row["alert_candle"]: continue
        if idx + 1 >= len(df_opt): continue 
            
        next_high = df_opt.iloc[idx + 1]["high"]
        if next_high > entry_price:
            candidates.append({
                "strike": strike,
                "diff": abs(entry_price - 150),
                "df": df_opt
            })
            
    if candidates:
        candidates.sort(key=lambda x: x["diff"])
        best = candidates[0]
        return True, best["strike"], best["df"]
    
    return False, None, None

def sequential_trades_with_validation(df, api_obj, master_df, expiry_date, trade_date):
    trades = []
    trade_open = False
    trade_entry = {}
    trade_count = 0
    last_trade_type = None 
    active_opt_df = None 
    
    for idx in range(len(df) - 1):
        if trade_count >= 2: break
        curr_time = df.loc[idx, "datetime"]

        # ENTRY
        if not trade_open and df.loc[idx, "alert_candle"]:
            # PE
            if df.loc[idx, "above_ema_alert"]:
                if last_trade_type != "PE":
                    if df.loc[idx + 1, "low"] < df.loc[idx, "low"]:
                        is_valid, best_strike, opt_df = validate_signal_with_options(
                            api_obj, master_df, expiry_date, curr_time, "PE", df.loc[idx, "open"], trade_date)
                        if is_valid:
                            trade_open = True
                            last_trade_type = "PE"
                            active_opt_df = opt_df 
                            opt_signal_row = opt_df[opt_df["datetime"] == curr_time].iloc[0]
                            opt_entry_price = opt_signal_row["high"]
                            opt_sl_price = opt_signal_row["low"]
                            risk = opt_entry_price - opt_sl_price
                            opt_target_price = opt_entry_price + (3 * risk)
                            trade_entry = {
                                "TradeID": trade_count + 1, "Nature": "BUY PE", "Signal Time": curr_time.strftime('%H:%M'), 
                                "Signal Close": df.loc[idx, "close"], "type": "PE", "entry_time": df.loc[idx + 1, "datetime"],
                                "entry_price": opt_entry_price, "SL": round(opt_sl_price, 2), "target": round(opt_target_price, 2),
                                "result": "OPEN", "Best Strike": best_strike, "SL Time": "-", "Target Time": "-"
                            }
                            trades.append(trade_entry) 

            # CE
            elif df.loc[idx, "below_ema_alert"]:
                if last_trade_type != "CE":
                    if df.loc[idx + 1, "high"] > df.loc[idx, "high"]:
                        is_valid, best_strike, opt_df = validate_signal_with_options(
                            api_obj, master_df, expiry_date, curr_time, "CE", df.loc[idx, "open"], trade_date)
                        if is_valid:
                            trade_open = True
                            last_trade_type = "CE"
                            active_opt_df = opt_df
                            opt_signal_row = opt_df[opt_df["datetime"] == curr_time].iloc[0]
                            opt_entry_price = opt_signal_row["high"]
                            opt_sl_price = opt_signal_row["low"]
                            risk = opt_entry_price - opt_sl_price
                            opt_target_price = opt_entry_price + (3 * risk)
                            trade_entry = {
                                "TradeID": trade_count + 1, "Nature": "BUY CE", "Signal Time": curr_time.strftime('%H:%M'),
                                "Signal Close": df.loc[idx, "close"], "type": "CE", "entry_time": df.loc[idx + 1, "datetime"],
                                "entry_price": opt_entry_price, "SL": round(opt_sl_price, 2), "target": round(opt_target_price, 2),
                                "result": "OPEN", "Best Strike": best_strike, "SL Time": "-", "Target Time": "-"
                            }
                            trades.append(trade_entry)

        # EXIT
        elif trade_open and active_opt_df is not None:
            opt_rows = active_opt_df[active_opt_df["datetime"] == curr_time]
            if not opt_rows.empty:
                opt_row = opt_rows.iloc[0]
                if opt_row["high"] >= trade_entry["target"]:
                    trade_entry.update({
                        "exit_time": curr_time, "exit_price": trade_entry["target"], 
                        "result": "TARGET", "Target Time": curr_time.strftime('%H:%M')
                    })
                    trade_open = False
                    trade_count += 1
                    break 
                elif opt_row["low"] <= trade_entry["SL"]:
                    trade_entry.update({
                        "exit_time": curr_time, "exit_price": trade_entry["SL"], 
                        "result": "SL", "SL Time": curr_time.strftime('%H:%M')
                    })
                    trade_open = False
                    trade_count += 1
    return trades

def plot_strategy_with_trades(df, trades=None, title="Chart"):
    fig = go.Figure(data=[go.Candlestick(
        x=df["datetime"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Candles"
    )])
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["ema_3_smooth"], mode="lines", 
                             line=dict(color='yellow', width=1), name="EMA Smooth"))

    if "alert_candle" in df.columns:
        alert_df = df[df["alert_candle"]]
        if not alert_df.empty:
            fig.add_trace(go.Scatter(x=alert_df["datetime"], y=alert_df["close"], mode="markers",
                                     marker=dict(size=6, symbol="diamond", color="white"), name="Alert Logic"))
    if trades:
        for tr in trades:
            dt_obj = datetime.strptime(f"{df['datetime'].iloc[0].date()} {tr['Signal Time']}", "%Y-%m-%d %H:%M")
            fig.add_trace(go.Scatter(
                x=[dt_obj], y=[tr["Signal Close"]], mode="markers",
                marker=dict(size=12, color="blue", symbol="triangle-down"), name="Signal"
            ))
            entry_time = tr["entry_time"]
            row_match = df[df["datetime"] == entry_time]
            if not row_match.empty:
                y_val = row_match.iloc[0]["open"] 
                label = f"{tr['type']} (Opt: {round(tr['entry_price'],1)})"
                fig.add_trace(go.Scatter(
                    x=[entry_time], y=[y_val], mode="markers+text",
                    marker=dict(size=10, color="orange", symbol="circle"), 
                    text=[label], textposition="bottom right", name="Entry"
                ))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# ---------------------------
# MAIN APP
# ---------------------------
api = angel_login()
if not api: st.stop()

master_df = get_scrip_master_df()
st.title("NIFTY Strategy (Seamless Live)")

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    chosen_date = st.date_input("Backtest Date", value=pd.Timestamp.today(), key="bd", on_change=check_and_fix_weekend, args=("bd",))
with col2:
    expiry_str = st.text_input("Strategy Expiry", value=get_next_tuesday(chosen_date))
with col3:
    start_btn = st.button("Start Live Monitor")
    stop_btn = st.button("Stop Monitor")

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "alerted_trades" not in st.session_state:
    st.session_state.alerted_trades = set() 
if "scan_initialized" not in st.session_state:
    st.session_state.scan_initialized = False

if start_btn:
    st.session_state.monitoring = True
    st.session_state.scan_initialized = False 
    st.session_state.alerted_trades = set() 
    
if stop_btn:
    st.session_state.monitoring = False

# --- PLACEHOLDERS ---
status_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

if st.session_state.monitoring:
    last_candle_time = None
    
    while st.session_state.monitoring:
        status_placeholder.caption(f"Status: 🟢 RUNNING | Last Scan: {datetime.now().strftime('%H:%M:%S')}")

        df_static = fetch_candle_data(api, NIFTY_TOKEN, "TEN_MINUTE", exchange="NSE", specific_date=chosen_date)
        
        if not df_static.empty:
            df_proc = add_custom_ema(df_static)
            trades = sequential_trades_with_validation(df_proc, api, master_df, expiry_str, chosen_date)
            
            # --- ON SCREEN ALERT LOGIC ---
            if trades:
                if not st.session_state.scan_initialized:
                    for t in trades:
                        st.session_state.alerted_trades.add(t["TradeID"])
                    st.session_state.scan_initialized = True
                else:
                    for t in trades:
                        if t["TradeID"] not in st.session_state.alerted_trades:
                            # 🚀 FOUND NEW OPPORTUNITY (Toast only)
                            st.toast(f"🚀 New Signal: {t['Nature']} @ {t['entry_price']}", icon="✅")
                            st.session_state.alerted_trades.add(t["TradeID"])

            # --- UPDATE UI ---
            with chart_placeholder.container():
                # Uses unique key based on timestamp to avoid duplicate ID errors
                st.plotly_chart(
                    plot_strategy_with_trades(df_proc, trades=trades, title=f"NIFTY Spot - {chosen_date}"), 
                    use_container_width=True, 
                    key=f"live_chart_{int(time.time())}"
                )
            
            with table_placeholder.container():
                if trades:
                    st.subheader("Detected Trades")
                    df_display = pd.DataFrame(trades)
                    st.dataframe(
                        df_display[["TradeID", "Nature", "Signal Time", "entry_price", "SL", "target", "SL Time", "Target Time", "result", "Best Strike"]].style.applymap(
                            lambda x: 'background-color: #d4edda; color: green' if x == 'TARGET' else ('background-color: #f8d7da; color: red' if x == 'SL' else ''), subset=['result']
                        ),
                        use_container_width=True
                    )
                else:
                    st.info("No trades detected yet.")
        
        else:
            status_placeholder.error("Error: Could not fetch Nifty data.")

        time.sleep(REFRESH_RATE_SEC)
else:
    status_placeholder.caption("Status: 🔴 STOPPED")