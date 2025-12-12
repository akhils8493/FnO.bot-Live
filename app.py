import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta, time as dtime
from SmartApi import SmartConnect
import requests
import os
import pyotp
import time
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------------------
# TIMEZONE SETUP
# ---------------------------
IST = pytz.timezone("Asia/Kolkata")

# ---------------------------
# CONFIGURATION
# ---------------------------
API_KEY = "WM6i3ikL"
CLIENT_ID = "AABY364105"
PIN = "6954"
TOTP_TOKEN = "D5PMGU3B674K4YFIQNE7CKDUSU"
NIFTY_TOKEN = "99926000"

# STRATEGY SETTINGS
MIN_OPT_PRICE = 150
MAX_OPT_PRICE = 190

# ---------------------------
# EMAIL CONFIGURATION
# ---------------------------
GMAIL_USER = "akhils8493@gmail.com"      
GMAIL_PASSWORD = "tptr wtof dhkb jtht" 

TO_EMAILS = [
    "akhils8493@gmail.com", 
    "shauryamraghaw@gmail.com",
    "kamal.padha99@gmail.com"
]

st.set_page_config(page_title="Nifty Auto-Bot", layout="wide")

# ---------------------------
# ANGEL LOGIN
# ---------------------------
@st.cache_resource(ttl=3600)
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
        date = datetime.now(IST).date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    days_ahead = (1 - date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_tues = date + timedelta(days=days_ahead)
    return next_tues.strftime("%d%b%Y").upper()

def get_atm_strike(price):
    return round(price / 50) * 50

# ---------------------------
# EMAIL FUNCTION
# ---------------------------
def send_email_notification(trade_details, alert_type="ENTRY"):
    try:
        if "TARGET" in str(trade_details['result']):
            subject_prefix = "✅ TARGET HIT"
        elif "SL" in str(trade_details['result']):
            subject_prefix = "🔴 SL HIT"
        else:
            subject_prefix = "🚀 NEW ENTRY"

        subject = f"{subject_prefix}: {trade_details['Nature']} @ {trade_details['entry_price']}"
        
        body = f"""
        🔔 TRADE UPDATE: {subject_prefix}
        
        ------------------------------------
        STATUS      : {trade_details['result']}
        ------------------------------------
        TYPE        : {trade_details['Nature']}
        STRIKE      : {trade_details['Best Strike']}
        ENTRY PRICE : {trade_details['entry_price']}
        TARGET      : {trade_details['target']}
        STOP LOSS   : {trade_details['SL']}
        TIME        : {datetime.now(IST).strftime('%H:%M:%S')}
        ------------------------------------
        
        *This is an automated alert from your Nifty Bot.*
        """

        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = ", ".join(TO_EMAILS)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        server.sendmail(GMAIL_USER, TO_EMAILS, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email Error: {e}")
        return False

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
    except Exception as e:
        return None

# ---------------------------
# DATA FETCHING
# ---------------------------
def fetch_ltp(api_obj, token, exchange="NSE"):
    try:
        data = api_obj.ltpData(exchange, token, token)
        if data and data.get("status"):
             return float(data["data"]["ltp"])
    except:
        pass
    return None

def fetch_candle_data(api_obj, token, interval, exchange="NSE", specific_date=None, days_back=5):
    try:
        now_ist = datetime.now(IST)
        target_date = specific_date if specific_date else now_ist.date()

        if target_date == now_ist.date():
            to_dt = now_ist
        else:
            to_dt = datetime.combine(target_date, dtime(15, 30))
            to_dt = IST.localize(to_dt)

        from_dt = to_dt - timedelta(days=days_back)
        from_dt = from_dt.replace(hour=9, minute=15)

        params = {
            "exchange": exchange, 
            "symboltoken": str(token), 
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"), 
            "todate": to_dt.strftime("%Y-%m-%d %H:%M")
        }
        
        for attempt in range(3):
            try:
                time.sleep(0.2) 
                resp = api_obj.getCandleData(params)
                if resp and resp.get("status") and resp.get("data"):
                    data = resp.get("data", [])
                    cols = ["datetime", "open", "high", "low", "close", "volume"]
                    df = pd.DataFrame(data, columns=cols)
                    
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    if df["datetime"].dt.tz is not None:
                        df["datetime"] = df["datetime"].dt.tz_localize(None)
                    
                    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
                    return df
                time.sleep(0.5)
            except:
                time.sleep(1)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def inject_live_ltp(df, api_obj, token, exchange="NSE"):
    if df.empty: return df
    
    now_ist = datetime.now(IST)
    if now_ist.time() >= dtime(15, 30): return df 
    
    ltp = fetch_ltp(api_obj, token, exchange)
    if ltp is None: return df 
    
    last_dt = df.iloc[-1]["datetime"].replace(tzinfo=None)
    now_naive = now_ist.replace(microsecond=0, tzinfo=None)
    
    minute_block = (now_naive.minute // 10) * 10
    current_candle_start = now_naive.replace(minute=minute_block, second=0)
    
    if last_dt == current_candle_start:
        df.at[df.index[-1], 'close'] = ltp
        df.at[df.index[-1], 'high'] = max(df.at[df.index[-1], 'high'], ltp)
        df.at[df.index[-1], 'low'] = min(df.at[df.index[-1], 'low'], ltp)
        
    elif last_dt < current_candle_start:
        new_row = pd.DataFrame([{
            "datetime": current_candle_start, "open": ltp, "high": ltp, "low": ltp, "close": ltp, "volume": 0
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    return df

# ---------------------------
# INDICATORS
# ---------------------------
def add_custom_ema(df):
    df = df.copy()
    df["ema_3_base"] = df["close"].ewm(span=3, adjust=False).mean()
    df["ema_3_smooth"] = df["ema_3_base"].shift(2)
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
            
        df_opt = fetch_candle_data(api_obj, token, "TEN_MINUTE", exchange="NFO", specific_date=trade_date, days_back=5)
        
        if trade_date == datetime.now(IST).date():
            df_opt = inject_live_ltp(df_opt, api_obj, token, exchange="NFO")
            
        if df_opt.empty: continue
        df_opt = add_custom_ema(df_opt)
        
        df_opt_today = df_opt[df_opt["datetime"].dt.date == trade_date].copy()
        df_opt_today = df_opt_today.reset_index(drop=True)
        
        match_rows = df_opt_today[df_opt_today["datetime"] == signal_time]
        if match_rows.empty: continue
            
        idx = match_rows.index[0]
        row = df_opt_today.iloc[idx]
        entry_price = row["high"]
        
        if not (MIN_OPT_PRICE <= entry_price <= MAX_OPT_PRICE): continue
        if not row["alert_candle"]: continue
        if idx + 1 >= len(df_opt_today): continue 
            
        if df_opt_today.iloc[idx + 1]["high"] > entry_price:
            candidates.append({
                "strike": strike, "diff": abs(entry_price - 150), "df": df_opt_today 
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

        if not trade_open and df.loc[idx, "alert_candle"]:
            if df.loc[idx, "above_ema_alert"]:
                if last_trade_type != "PE":
                    if df.loc[idx + 1, "low"] < df.loc[idx, "low"]:
                        is_valid, best_strike, opt_df = validate_signal_with_options(
                            api_obj, master_df, expiry_date, curr_time, "PE", df.loc[idx, "open"], trade_date)
                        if is_valid:
                            trade_open = True
                            last_trade_type = "PE"
                            active_opt_df = opt_df 
                            opt_alert_row = opt_df[opt_df["datetime"] == curr_time].iloc[0]
                            opt_entry_price = opt_alert_row["high"]
                            final_sl = opt_alert_row["low"]
                            risk = opt_entry_price - final_sl
                            opt_target_price = opt_entry_price + (3 * risk)
                            trade_entry = {
                                "TradeID": trade_count + 1, "Nature": "BUY PE", "Signal Time": curr_time.strftime('%H:%M'), 
                                "Signal Close": df.loc[idx, "close"], "type": "PE", "entry_time": df.loc[idx + 1, "datetime"],
                                "entry_price": opt_entry_price, "SL": round(final_sl, 2), "target": round(opt_target_price, 2),
                                "result": "OPEN ⏳", "Best Strike": best_strike, "SL Time": "-", "Target Time": "-"
                            }
            elif df.loc[idx, "below_ema_alert"]:
                if last_trade_type != "CE":
                    if df.loc[idx + 1, "high"] > df.loc[idx, "high"]:
                        is_valid, best_strike, opt_df = validate_signal_with_options(
                            api_obj, master_df, expiry_date, curr_time, "CE", df.loc[idx, "open"], trade_date)
                        if is_valid:
                            trade_open = True
                            last_trade_type = "CE"
                            active_opt_df = opt_df
                            opt_alert_row = opt_df[opt_df["datetime"] == curr_time].iloc[0]
                            opt_entry_price = opt_alert_row["high"]
                            final_sl = opt_alert_row["low"]
                            risk = opt_entry_price - final_sl
                            opt_target_price = opt_entry_price + (3 * risk)
                            trade_entry = {
                                "TradeID": trade_count + 1, "Nature": "BUY CE", "Signal Time": curr_time.strftime('%H:%M'),
                                "Signal Close": df.loc[idx, "close"], "type": "CE", "entry_time": df.loc[idx + 1, "datetime"],
                                "entry_price": opt_entry_price, "SL": round(final_sl, 2), "target": round(opt_target_price, 2),
                                "result": "OPEN ⏳", "Best Strike": best_strike, "SL Time": "-", "Target Time": "-"
                            }

        elif trade_open and active_opt_df is not None:
            opt_rows = active_opt_df[active_opt_df["datetime"] == curr_time]
            if not opt_rows.empty:
                opt_row = opt_rows.iloc[0]
                if opt_row["high"] >= trade_entry["target"]:
                    trade_entry.update({"exit_time": curr_time, "exit_price": trade_entry["target"], "result": "TARGET 🟢", "Target Time": curr_time.strftime('%H:%M')})
                    trades.append(trade_entry) 
                    trade_open = False
                    trade_count += 1
                    break 
                elif opt_row["low"] <= trade_entry["SL"]:
                    trade_entry.update({"exit_time": curr_time, "exit_price": trade_entry["SL"], "result": "SL 🔴", "SL Time": curr_time.strftime('%H:%M')})
                    trades.append(trade_entry) 
                    trade_open = False
                    trade_count += 1
    
    if trade_open and trade_entry: trades.append(trade_entry)
    return trades

def plot_strategy_with_trades(df, trades=None, title="Chart", extra_lines=None, is_option_chart=False):
    fig = go.Figure(data=[go.Candlestick(x=df["datetime"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Candles")])
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["ema_3_smooth"], mode="lines", line=dict(color='yellow', width=1), name="EMA 3 (Offset 2)"))
    
    if "alert_candle" in df.columns:
        alert_df = df[df["alert_candle"]]
        if not alert_df.empty:
            fig.add_trace(go.Scatter(x=alert_df["datetime"], y=alert_df["close"], mode="markers", marker=dict(size=6, symbol="diamond", color="white"), name="Alert Logic"))
            
    if trades:
        for tr in trades:
            if not is_option_chart:
                sig_time_str = tr['Signal Time']
                sig_row = df[df['datetime'].dt.strftime('%H:%M') == sig_time_str]
                if not sig_row.empty:
                    fig.add_trace(go.Scatter(x=[sig_row.iloc[0]['datetime']], y=[tr["Signal Close"]], mode="markers", marker=dict(size=12, color="blue", symbol="triangle-down"), name="Signal"))
            
            entry_t = tr.get("entry_time")
            if entry_t:
                entry_row = df[df["datetime"] == entry_t]
                if not entry_row.empty:
                    fig.add_trace(go.Scatter(x=[entry_t], y=[entry_row.iloc[0]["open"]], mode="markers+text", marker=dict(size=10, color="orange"), text=[f"{tr['type']} Entry"], textposition="bottom right", name="Entry"))

    if extra_lines:
        if extra_lines.get("entry"): fig.add_hline(y=extra_lines["entry"], line_dash="dash", line_color="blue", annotation_text="ENTRY")
        if extra_lines.get("sl"): fig.add_hline(y=extra_lines["sl"], line_dash="solid", line_color="red", annotation_text="SL")
        if extra_lines.get("target"): fig.add_hline(y=extra_lines["target"], line_dash="solid", line_color="green", annotation_text="TARGET")

    fig.update_layout(title=title, template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    return fig

# ---------------------------
# MAIN APP
# ---------------------------
api = angel_login()
if not api: st.stop()
master_df = get_scrip_master_df()

with st.sidebar:
    st.header("⚙️ Controls")
    today_ist = datetime.now(IST).date()
    mode = st.radio("Select Mode", ["🔙 BACKTEST", "🔴 LIVE MARKET"], index=1)
    chosen_date = st.date_input("Date", value=today_ist) if mode == "🔙 BACKTEST" else today_ist
    expiry_str = st.text_input("Expiry", value=get_next_tuesday(chosen_date))
    refresh_rate = st.slider("Auto-Refresh (Sec)", 5, 60, 5)

    st.divider()
    if st.button("Send Test Alert"):
        if send_email_notification({"Nature": "TEST - BUY CE", "Best Strike": 24500, "Signal Time": "NOW", "entry_price": 100, "target": 120, "SL": 80, "result": "TEST"}):
            st.success("Sent!")
        else: st.error("Failed")

st.title("🚀 Nifty Master Bot (Sequential)")
top_status = st.empty() 
main_chart = st.empty() 
main_table = st.empty() 
analyzer_section = st.empty()

if "trade_state" not in st.session_state:
    st.session_state["trade_state"] = {}

# ---------------------------
# CORE STATE CHECK & RUN LOGIC
# ---------------------------
def is_recent(timestamp_str, limit_minutes=15):
    try:
        now = datetime.now(IST)
        event_time = datetime.strptime(timestamp_str, '%H:%M').time()
        event_dt = datetime.combine(now.date(), event_time)
        event_dt = IST.localize(event_dt) 
        diff = now - event_dt
        return diff < timedelta(minutes=limit_minutes) and diff >= timedelta(seconds=0)
    except:
        return True 

def run_analysis_cycle():
    df_long = fetch_candle_data(api, NIFTY_TOKEN, "TEN_MINUTE", exchange="NSE", specific_date=chosen_date, days_back=5)
    
    if mode == "🔴 LIVE MARKET":
        df_long = inject_live_ltp(df_long, api, NIFTY_TOKEN, exchange="NSE")

    if df_long.empty:
        main_chart.warning(f"⚠️ No Data. Time: {datetime.now(IST).strftime('%H:%M:%S')}")
        return
        
    df_long = add_custom_ema(df_long)
    df_today = df_long[df_long['datetime'].dt.date == chosen_date].copy()
    df_today = df_today.reset_index(drop=True)

    if df_today.empty:
        main_chart.warning(f"⚠️ No Data for Today. Time: {datetime.now(IST).strftime('%H:%M:%S')}")
        return
    
    trades = sequential_trades_with_validation(df_today, api, master_df, expiry_str, chosen_date)
    
    # 1. Main Spot Chart
    main_chart.plotly_chart(plot_strategy_with_trades(df_today, trades=trades, title=f"NIFTY - {datetime.now(IST).strftime('%H:%M:%S')}", is_option_chart=False), use_container_width=True)
    
    if trades:
        df_display = pd.DataFrame(trades)
        styler = df_display[["TradeID", "Nature", "Signal Time", "entry_price", "SL", "target", "result", "Best Strike", "SL Time", "Target Time"]].style.map(lambda v: 'background-color: #006400' if 'TARGET' in str(v) else ('background-color: #8B0000' if 'SL' in str(v) else 'background-color: #00008B'), subset=['result'])
        main_table.dataframe(styler, use_container_width=True)
        
        # ----------------------------------
        # 🔎 SEQUENTIAL OPTION CHARTS (BACKTEST ONLY)
        # ----------------------------------
        if mode == "🔙 BACKTEST":
            with analyzer_section.container():
                st.divider()
                st.subheader("🔎 Trade Analysis (All Trades)")
                for t_data in trades:
                    strike = t_data['Best Strike']
                    op_type = "CE" if "CE" in t_data['Nature'] else "PE"
                    st.write(f"### Trade #{t_data['TradeID']} - {t_data['Nature']} ({strike})")
                    
                    token = get_token_from_master_cached(master_df, "NIFTY", expiry_str, strike, op_type)
                    if token:
                        df_opt_chart = fetch_candle_data(api, token, "TEN_MINUTE", exchange="NFO", specific_date=chosen_date, days_back=5)
                        if not df_opt_chart.empty:
                            df_opt_chart = add_custom_ema(df_opt_chart)
                            df_opt_today_chart = df_opt_chart[df_opt_chart["datetime"].dt.date == chosen_date].copy()
                            lines = {"entry": t_data['entry_price'], "sl": t_data['SL'], "target": t_data['target']}
                            fig_opt = plot_strategy_with_trades(df_opt_today_chart, trades=[t_data], title=f"OPTION: {strike} {op_type} ({t_data['result']})", extra_lines=lines, is_option_chart=True)
                            st.plotly_chart(fig_opt, use_container_width=True)
                        else: st.error(f"No Data for {strike} {op_type}")
                    st.divider()

        # ----------------------------------
        # LIVE NOTIFICATIONS
        # ----------------------------------
        if mode == "🔴 LIVE MARKET":
            analyzer_section.empty()
            for trade in trades:
                tid = trade["TradeID"]
                current_result = trade["result"]
                signal_time_str = trade["Signal Time"]
                last_known_state = st.session_state["trade_state"].get(tid, None)
                is_fresh_event = is_recent(signal_time_str, limit_minutes=15)

                if last_known_state is None:
                    if is_fresh_event:
                        st.toast(f"New Entry: #{tid}", icon="🚀")
                        if send_email_notification(trade, alert_type="ENTRY"):
                            st.session_state["trade_state"][tid] = current_result
                    else:
                        st.session_state["trade_state"][tid] = current_result
                        
                elif last_known_state != current_result:
                    if "OPEN" in last_known_state and ("TARGET" in current_result or "SL" in current_result):
                        exit_time_str = trade.get("Target Time", "-") if "TARGET" in current_result else trade.get("SL Time", "-")
                        is_fresh_exit = is_recent(exit_time_str, limit_minutes=15) if exit_time_str != "-" else True

                        if is_fresh_exit:
                            st.toast(f"Exit: #{tid} ({current_result})", icon="🔔")
                            if send_email_notification(trade, alert_type="EXIT"):
                                st.session_state["trade_state"][tid] = current_result
                        else:
                            st.session_state["trade_state"][tid] = current_result
    else:
        main_table.info("⏳ Waiting for Signals...")
        
    top_status.caption(f"Last Scan: {datetime.now(IST).strftime('%H:%M:%S')}")

# ----------------------------------
# EXECUTION LOGIC (AUTO-START)
# ----------------------------------
if mode == "🔙 BACKTEST":
    if "backtest_data" not in st.session_state:
        st.session_state["backtest_data"] = None

    if st.button("Run Backtest"):
        with st.spinner("Analyzing..."):
            run_analysis_cycle()
            st.session_state["backtest_data"] = True

elif mode == "🔴 LIVE MARKET":
    # AUTO-START LOGIC
    # No "Start Button" needed. We assume if you selected "LIVE", you want it running.
    
    # 1. Provide a Stop Button
    if "stop_bot" not in st.session_state:
        st.session_state.stop_bot = False

    if st.sidebar.button("⏹ STOP BOT"):
        st.session_state.stop_bot = True
    
    if st.sidebar.button("▶️ RESTART"):
        st.session_state.stop_bot = False

    # 2. Check if we should run
    if not st.session_state.stop_bot:
        st.sidebar.success("✅ Bot Running...")
        
        # 3. Check Market Hours for Auto-Stop
        now_time = datetime.now(IST).time()
        market_open = dtime(9, 15) <= now_time <= dtime(15, 30)
        
        if market_open:
            # Loop for Live Trading
            while True:
                try:
                    run_analysis_cycle()
                    time.sleep(refresh_rate)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    top_status.error(f"Loop Error: {e}")
                    time.sleep(10)
        else:
            # Market Closed: Run Once (Static View) then Stop
            run_analysis_cycle()
            st.info("Market Closed. Bot is in Sleep Mode.")
    else:
        st.warning("Bot Manually Stopped.")
