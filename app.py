import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta, time as dtime
from SmartApi import SmartConnect
from dhanhq import dhanhq  # NEW IMPORT
import requests
import os
import pyotp
import time
import pytz
import smtplib
import math
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------------------
# SSL CONTEXT (For Dhan CSV)
# ---------------------------
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------
# TIMEZONE SETUP
# ---------------------------
IST = pytz.timezone("Asia/Kolkata")

# ---------------------------
# CONFIGURATION
# ---------------------------
# --- ANGEL ONE CREDENTIALS ---
API_KEY = "WM6i3ikL"
CLIENT_ID = "AABY364105"
PIN = "6954"
TOTP_TOKEN = "D5PMGU3B674K4YFIQNE7CKDUSU"
NIFTY_TOKEN = "99926000"

# --- DHAN CREDENTIALS (REPLACE THESE) ---
DHAN_CLIENT_ID = "2512259667"   # e.g., "1100087829"
DHAN_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY2NzQ0MTc4LCJpYXQiOjE3NjY2NTc3NzgsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAwMDg3ODI5In0.GAaZBKkdn6Fay6OOQQiMdIhIbV6d5tZsBfQCLMdHgg4EkAjTFKeIPCJPb1J7LXp0gc78ryOPahH8EbyAnXYnYQ" # Long JWT string


# STRATEGY SETTINGS
MIN_OPT_PRICE = 142
MAX_OPT_PRICE = 195
ORDER_QTY = 75  # Fixed Quantity (1 Lot)

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

st.set_page_config(page_title="Nifty Angel+Dhan Bot", layout="wide")

# ---------------------------
# 1. ANGEL ONE LOGIN
# ---------------------------
@st.cache_resource(ttl=3600)
def angel_login():
    try:
        obj = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_TOKEN).now()
        data = obj.generateSession(CLIENT_ID, PIN, totp)
        if not data or not data.get("status"):
            st.error(f"Angel Login failed: {data.get('message') if data else 'No response'}")
            return None
        return obj
    except Exception as e:
        st.error(f"Angel login error: {e}")
        return None

# ---------------------------
# 2. DHAN LOGIN
# ---------------------------
@st.cache_resource(ttl=3600)
def dhan_login():
    try:
        dhan = dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
        # Verify connection by checking funds
        funds = dhan.get_fund_limits()
        if funds.get("status") == "success":
            return dhan
        else:
            st.error(f"Dhan Login Failed. Check Token. Response: {funds}")
            return None
    except Exception as e:
        st.error(f"Dhan Connection Error: {e}")
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
    next_tues = date + timedelta(days=days_ahead)
    return next_tues.strftime("%d%b%Y").upper()

def get_atm_strike(price):
    return round(price / 50) * 50

# ---------------------------
# EMAIL FUNCTION
# ---------------------------
def send_email_notification(data, alert_type="ENTRY"):
    try:
        if alert_type == "ORDER":
            subject = f"✅ ORDER PLACED: {data['Symbol']} (ID: {data['OrderID']})"
            body = f"""
            🚀 ORDER EXECUTION REPORT
            
            ------------------------------------
            SYMBOL      : {data['Symbol']}
            ORDER ID    : {data['OrderID']}
            ------------------------------------
            TYPE        : {data['Type']}
            QUANTITY    : {data['Qty']}
            PRICE       : {data['Price']}
            TOTAL VAL   : {data['Total Amount']}
            TIME        : {data['Time']}
            ------------------------------------
            """
        else:
            if "TARGET" in str(data['result']):
                subject_prefix = "✅ TARGET HIT"
            elif "SL" in str(data['result']):
                subject_prefix = "🔴 SL HIT"
            else:
                subject_prefix = "🚀 NEW SIGNAL"

            time_display = data.get('Signal Time', datetime.now(IST).strftime('%H:%M:%S'))
            subject = f"{subject_prefix}: {data['Nature']} @ {data['entry_price']}"
            
            body = f"""
            🔔 STRATEGY ALERT: {subject_prefix}
            
            ------------------------------------
            STATUS      : {data['result']}
            ------------------------------------
            TYPE        : {data['Nature']}
            STRIKE      : {data['Best Strike']}
            ENTRY PRICE : {data['entry_price']}
            TARGET      : {data['target']}
            STOP LOSS   : {data['SL']}
            TIME        : {time_display}
            ------------------------------------
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
# SCRIP MASTERS (ANGEL & DHAN)
# ---------------------------
@st.cache_data(ttl=3600)
def get_angel_master_df():
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

@st.cache_data(ttl=3600)
def get_dhan_master_df():
    try:
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        df = pd.read_csv(url, low_memory=False)
        df.columns = df.columns.str.strip()
        # Parse date for filtering efficiency
        df["SEM_EXPIRY_DATE"] = pd.to_datetime(df["SEM_EXPIRY_DATE"], errors='coerce').dt.date
        return df
    except Exception as e:
        st.error(f"Error fetching Dhan Master: {e}")
        return pd.DataFrame()

def get_angel_token(df_master, symbol, expiry_date, strike, option_type):
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

def get_dhan_security_id(dhan_master, expiry_str, strike, option_type):
    """
    Converts Angel format Expiry (e.g. 30DEC2025) to Dhan format and finds Security ID
    """
    try:
        target_date = datetime.strptime(expiry_str, "%d%b%Y").date()
        
        filtered = dhan_master[
            (dhan_master["SEM_EXM_EXCH_ID"] == "NSE") &
            (dhan_master["SEM_INSTRUMENT_NAME"] == "OPTIDX") &
            (dhan_master["SEM_TRADING_SYMBOL"].str.contains("NIFTY")) &
            (~dhan_master["SEM_TRADING_SYMBOL"].str.contains("FINNIFTY")) &
            (dhan_master["SEM_EXPIRY_DATE"] == target_date) &
            (dhan_master["SEM_STRIKE_PRICE"] == float(strike)) &
            (dhan_master["SEM_OPTION_TYPE"] == option_type)
        ]

        if not filtered.empty:
            return str(filtered.iloc[0]["SEM_SMST_SECURITY_ID"])
        return None
    except Exception as e:
        print(f"Dhan Token Lookup Error: {e}")
        return None

def get_symbol_from_token(df_master, token):
    try:
        row = df_master[df_master['token'] == str(token)]
        if not row.empty:
            return row.iloc[0]['symbol']
        return None
    except:
        return None

# ---------------------------
# ORDER PLACEMENT (ANGEL + DHAN)
# ---------------------------
def place_angel_order(api_obj, token, symbol, price):
    try:
        orderparams = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": str(token),
            "transactiontype": "BUY",
            "exchange": "NFO",
            "ordertype": "LIMIT",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": str(price),
            "quantity": str(ORDER_QTY)
        }
        orderId = api_obj.placeOrder(orderparams)
        return orderId
    except Exception as e:
        print(f"Angel Order Error: {e}")
        return False

def place_dhan_order(dhan_obj, security_id, price):
    try:
        order = dhan_obj.place_order(
            security_id=security_id,
            exchange_segment="NSE_FNO",
            transaction_type="BUY",
            quantity=ORDER_QTY,
            order_type="LIMIT",
            product_type="INTRADAY",
            price=float(price),
            validity="DAY"
        )
        if order["status"] == "success":
            return order["data"]["orderId"]
        else:
            print(f"Dhan Order Failed: {order}")
            return False
    except Exception as e:
        print(f"Dhan Exec Error: {e}")
        return False

# ---------------------------
# DATA FETCHING (ANGEL)
# ---------------------------
def fetch_ltp(api_obj, token, exchange="NSE"):
    try:
        data = api_obj.ltpData(exchange, token, token)
        if data and data.get("status"):
             return float(data["data"]["ltp"])
    except:
        pass
    return None

def fetch_candle_data(api_obj, token, interval, exchange="NSE", specific_date=None, days_back=60, custom_from=None, custom_to=None):
    try:
        if custom_from and custom_to:
            from_dt = custom_from
            to_dt = custom_to
        else:
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
                time.sleep(0.1) 
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
                time.sleep(0.3)
            except:
                time.sleep(0.5)
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
# INDICATORS & LOGIC (BUFFERED)
# ---------------------------
def add_custom_ema(df):
    df = df.copy()
    df["ema_3_base"] = df["close"].ewm(span=3, adjust=False).mean()
    df["ema_3_smooth"] = df["ema_3_base"].shift(2)
    
    df["body_top"] = df[["open", "close"]].max(axis=1)
    df["body_bottom"] = df[["open", "close"]].min(axis=1)

    buffer = 2.0 

    df["above_ema_alert"] = df["body_bottom"] > (df["ema_3_smooth"] + buffer)
    df["below_ema_alert"] = df["body_top"] < (df["ema_3_smooth"] - buffer)
    df["alert_candle"] = df["above_ema_alert"] | df["below_ema_alert"]
    return df

def find_precise_event_time(api_obj, token, start_time_10min, threshold_price, condition="GT", exchange="NSE"):
    try:
        from_dt = start_time_10min
        to_dt = start_time_10min + timedelta(minutes=10)
        
        df_1min = fetch_candle_data(api_obj, token, "ONE_MINUTE", exchange=exchange, custom_from=from_dt, custom_to=to_dt)
        
        if df_1min.empty:
            return start_time_10min 
            
        for _, row in df_1min.iterrows():
            if condition == "GT":
                if row['high'] >= threshold_price: 
                    return row['datetime'] 
            elif condition == "LT":
                if row['low'] <= threshold_price: 
                    return row['datetime']
        return start_time_10min 
    except:
        return start_time_10min

def validate_signal_with_options(api_obj, master_df, expiry_date, signal_time, signal_type, spot_price, trade_date):
    atm = get_atm_strike(spot_price)
    strikes = [atm + (i * 50) for i in range(-5, 6)]
    candidates = []
    
    for strike in strikes:
        token = get_angel_token(master_df, "NIFTY", expiry_date, strike, signal_type)
        if not token: continue
            
        df_opt = fetch_candle_data(api_obj, token, "TEN_MINUTE", exchange="NFO", specific_date=trade_date, days_back=60)
        
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
                "strike": strike, "diff": abs(entry_price - 150), "df": df_opt_today, "token": token
            })
            
    if candidates:
        candidates.sort(key=lambda x: x["diff"])
        best = candidates[0]
        return True, best["strike"], best["df"], best["token"]
    return False, None, None, None

def sequential_trades_with_validation(df, api_obj, master_df, expiry_date, trade_date, is_live_mode=False):
    trades = []
    trade_open = False
    trade_entry = {} 
    trade_count = 0
    last_trade_type = None 
    active_opt_df = None 
    active_opt_token = None 
    
    for idx in range(len(df) - 1):
        if trade_count >= 2: break
        
        curr_time = df.loc[idx, "datetime"] 
        entry_idx = idx + 1
        entry_candle_time = df.loc[entry_idx, "datetime"]
        
        is_current_live_candle = is_live_mode and (entry_idx == len(df) - 1)

        if not trade_open and df.loc[idx, "alert_candle"]:
            # BUY PE
            if df.loc[idx, "above_ema_alert"]:
                if last_trade_type != "PE":
                    if df.loc[entry_idx, "low"] < df.loc[idx, "low"]:
                        
                        is_valid, best_strike, opt_df, opt_token = validate_signal_with_options(
                            api_obj, master_df, expiry_date, curr_time, "PE", df.loc[idx, "open"], trade_date)
                        
                        if is_valid:
                            opt_alert_idx = opt_df[opt_df["datetime"] == curr_time].index[0]
                            opt_alert_row = opt_df.iloc[opt_alert_idx]
                            opt_entry_row = opt_df.iloc[opt_alert_idx + 1] 
                            opt_entry_price = opt_alert_row["high"]
                            
                            if is_current_live_candle:
                                display_time = datetime.now(IST).strftime('%H:%M:%S')
                            else:
                                precise_dt = find_precise_event_time(
                                    api_obj, opt_token, entry_candle_time, opt_entry_price, 
                                    condition="GT", exchange="NFO"
                                )
                                display_time = precise_dt.strftime('%H:%M')
                            
                            trade_open = True
                            last_trade_type = "PE"
                            active_opt_df = opt_df
                            active_opt_token = opt_token
                            
                            # --- CALCULATIONS ---
                            raw_sl_val = opt_alert_row["low"] 
                            if opt_entry_row["low"] < raw_sl_val: 
                                raw_sl_val = opt_entry_row["low"]

                            exact_risk = opt_entry_price - raw_sl_val
                            exact_target = opt_entry_price + (3 * exact_risk)

                            final_sl = math.floor(raw_sl_val)
                            final_target = math.floor(exact_target / 5) * 5

                            trade_entry = {
                                "TradeID": trade_count + 1, "Nature": "BUY PE", "Signal Time": display_time, 
                                "Signal Close": df.loc[idx, "close"], "type": "PE", "entry_time": entry_candle_time,
                                "entry_price": opt_entry_price, "SL": final_sl, "target": final_target,
                                "result": "OPEN ⏳", "Best Strike": best_strike, "SL Time": "-", "Target Time": "-",
                                "token": opt_token
                            }

            # BUY CE
            elif df.loc[idx, "below_ema_alert"]:
                if last_trade_type != "CE":
                    if df.loc[entry_idx, "high"] > df.loc[idx, "high"]:
                        
                        is_valid, best_strike, opt_df, opt_token = validate_signal_with_options(
                            api_obj, master_df, expiry_date, curr_time, "CE", df.loc[idx, "open"], trade_date)
                        
                        if is_valid:
                            opt_alert_idx = opt_df[opt_df["datetime"] == curr_time].index[0]
                            opt_alert_row = opt_df.iloc[opt_alert_idx]
                            opt_entry_row = opt_df.iloc[opt_alert_idx + 1] 
                            opt_entry_price = opt_alert_row["high"]

                            if is_current_live_candle:
                                display_time = datetime.now(IST).strftime('%H:%M:%S')
                            else:
                                precise_dt = find_precise_event_time(
                                    api_obj, opt_token, entry_candle_time, opt_entry_price, 
                                    condition="GT", exchange="NFO"
                                )
                                display_time = precise_dt.strftime('%H:%M')

                            trade_open = True
                            last_trade_type = "CE"
                            active_opt_df = opt_df
                            active_opt_token = opt_token

                            # --- CALCULATIONS ---
                            raw_sl_val = opt_alert_row["low"] 
                            if opt_entry_row["low"] < raw_sl_val: 
                                raw_sl_val = opt_entry_row["low"]

                            exact_risk = opt_entry_price - raw_sl_val
                            exact_target = opt_entry_price + (3 * exact_risk)

                            final_sl = math.floor(raw_sl_val)
                            final_target = math.floor(exact_target / 5) * 5

                            trade_entry = {
                                "TradeID": trade_count + 1, "Nature": "BUY CE", "Signal Time": display_time,
                                "Signal Close": df.loc[idx, "close"], "type": "CE", "entry_time": entry_candle_time,
                                "entry_price": opt_entry_price, "SL": final_sl, "target": final_target,
                                "result": "OPEN ⏳", "Best Strike": best_strike, "SL Time": "-", "Target Time": "-",
                                "token": opt_token
                            }

        elif trade_open and active_opt_df is not None:
            opt_rows = active_opt_df[active_opt_df["datetime"] == curr_time]
            if not opt_rows.empty:
                opt_row = opt_rows.iloc[0]
                
                # TARGET
                if opt_row["high"] >= trade_entry["target"]:
                    if is_current_live_candle:
                          exit_display_time = datetime.now(IST).strftime('%H:%M:%S')
                    else:
                        precise_dt = find_precise_event_time(
                            api_obj, active_opt_token, curr_time, trade_entry["target"], 
                            condition="GT", exchange="NFO"
                        )
                        exit_display_time = precise_dt.strftime('%H:%M')

                    trade_entry.update({"exit_time": curr_time, "exit_price": trade_entry["target"], "result": "TARGET 🟢", "Target Time": exit_display_time})
                    trades.append(trade_entry) 
                    trade_open = False
                    trade_count += 1
                    break 
                
                # SL
                elif opt_row["low"] <= trade_entry["SL"]:
                    is_entry_candle = (curr_time == trade_entry["entry_time"])
                    is_touching_sl = (opt_row["low"] == trade_entry["SL"])
                    
                    if is_entry_candle and is_touching_sl:
                        pass
                    else:
                        if is_current_live_candle:
                            exit_display_time = datetime.now(IST).strftime('%H:%M:%S')
                        else:
                            precise_dt = find_precise_event_time(
                                api_obj, active_opt_token, curr_time, trade_entry["SL"], 
                                condition="LT", exchange="NFO"
                            )
                            exit_display_time = precise_dt.strftime('%H:%M')

                        trade_entry.update({"exit_time": curr_time, "exit_price": trade_entry["SL"], "result": "SL 🔴", "SL Time": exit_display_time})
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
                entry_t = tr.get("entry_time")
                if entry_t:
                    entry_row = df[df["datetime"] == entry_t]
                    if not entry_row.empty:
                        fig.add_trace(go.Scatter(x=[entry_t], y=[tr["Signal Close"]], mode="markers", marker=dict(size=12, color="blue", symbol="triangle-down"), name="Signal"))
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
dhan_api = dhan_login() # DHAN Login

if not api:
    st.error("Angel One Login Failed. Stop.")
    st.stop()
if not dhan_api:
    st.error("Dhan Login Failed. Stop.")
    st.stop()

master_df = get_angel_master_df()
dhan_master_df = get_dhan_master_df()

with st.sidebar:
    st.header("⚙️ Controls")
    today_ist = datetime.now(IST).date()
    mode = st.radio("Select Mode", ["🔙 BACKTEST", "🔴 LIVE MARKET"], index=1)
    chosen_date = st.date_input("Date", value=today_ist) if mode == "🔙 BACKTEST" else today_ist
    expiry_str = st.text_input("Expiry", value=get_next_tuesday(chosen_date))
    
    st.caption(f"🗓️ Using Expiry: **{expiry_str}**")
    
    refresh_rate = st.slider("Auto-Refresh (Sec)", 1, 60, 2)

    st.divider()
    if st.button("Send Test Alert"):
        if send_email_notification({"Nature": "TEST - BUY CE", "Best Strike": 24500, "Signal Time": datetime.now(IST).strftime('%H:%M:%S'), "entry_price": 100, "target": 120, "SL": 80, "result": "TEST"}):
            st.success("Sent!")
        else: st.error("Failed")

st.title("🚀 Nifty Auto-Bot (Angel Data + Dhan Exec)")
top_status = st.empty() 
main_chart = st.empty() 
main_table = st.empty() 
order_table = st.empty() 
analyzer_section = st.empty()

# --- INITIALIZE SESSION STATE ---
if "trade_state" not in st.session_state:
    st.session_state["trade_state"] = {}

if "session_trades" not in st.session_state:
    st.session_state["session_trades"] = []

if "order_book" not in st.session_state:
    st.session_state["order_book"] = []

# ---------------------------
# CORE STATE CHECK & RUN LOGIC
# ---------------------------
def is_recent(timestamp_str, limit_minutes=15):
    try:
        now = datetime.now(IST)
        has_seconds = len(timestamp_str.split(':')) == 3
        fmt = '%H:%M:%S' if has_seconds else '%H:%M'
        
        event_time = datetime.strptime(timestamp_str, fmt).time()
        event_dt = datetime.combine(now.date(), event_time)
        event_dt = IST.localize(event_dt) 
        
        diff = now - event_dt
        return diff < timedelta(minutes=limit_minutes) and diff >= timedelta(seconds=0)
    except:
        return True 

def run_analysis_cycle():
    df_long = fetch_candle_data(
        api, 
        NIFTY_TOKEN, 
        "TEN_MINUTE", 
        exchange="NSE", 
        specific_date=chosen_date, 
        days_back=60 
    )
    
    is_live = False
    
    if mode == "🔴 LIVE MARKET":
        df_long = inject_live_ltp(df_long, api, NIFTY_TOKEN, exchange="NSE")
        is_live = True

    if df_long.empty:
        main_chart.warning(f"⚠️ No Data. Time: {datetime.now(IST).strftime('%H:%M:%S')}")
        return
        
    df_long = add_custom_ema(df_long)
    
    df_today = df_long[df_long['datetime'].dt.date == chosen_date].copy()
    df_today = df_today.reset_index(drop=True)

    if df_today.empty:
        main_chart.warning(f"⚠️ No Data for Today. Time: {datetime.now(IST).strftime('%H:%M:%S')}")
        return
    
    current_trades = sequential_trades_with_validation(df_today, api, master_df, expiry_str, chosen_date, is_live_mode=is_live)
    
    # --- SAFETY CHECK: State Continuity ---
    should_update = True
    
    if st.session_state["session_trades"]:
        last_saved_trade = st.session_state["session_trades"][-1]
        
        if "OPEN" in last_saved_trade["result"]:
            found_in_new = False
            for t in current_trades:
                if t["entry_time"] == last_saved_trade["entry_time"]:
                    found_in_new = True
                    break
            
            if not found_in_new:
                should_update = False
                st.toast("⚠️ Data Glitch Detected! Open Trade vanished. Keeping old state.", icon="🛡️")
                main_chart.warning(f"⚠️ Data Instability detected at {datetime.now(IST).strftime('%H:%M:%S')}. Ignoring scan.")

    if should_update:
        st.session_state["session_trades"] = current_trades
    
    trades_to_display = st.session_state["session_trades"]

    main_chart.plotly_chart(plot_strategy_with_trades(df_today, trades=trades_to_display, title=f"NIFTY - {datetime.now(IST).strftime('%H:%M:%S')}", is_option_chart=False), use_container_width=True)
    
    if trades_to_display:
        df_display = pd.DataFrame(trades_to_display)
        styler = df_display[["TradeID", "Nature", "Signal Time", "entry_price", "SL", "target", "result", "Best Strike", "SL Time", "Target Time"]].style.map(lambda v: 'background-color: #006400' if 'TARGET' in str(v) else ('background-color: #8B0000' if 'SL' in str(v) else 'background-color: #00008B'), subset=['result'])
        main_table.dataframe(styler, use_container_width=True)
        
        if mode == "🔙 BACKTEST":
            with analyzer_section.container():
                st.divider()
                st.subheader("🔎 Trade Analysis (All Trades)")
                for t_data in trades_to_display:
                    strike = t_data['Best Strike']
                    op_type = "CE" if "CE" in t_data['Nature'] else "PE"
                    st.write(f"### Trade #{t_data['TradeID']} - {t_data['Nature']} ({strike})")
                    
                    token = get_angel_token(master_df, "NIFTY", expiry_str, strike, op_type)
                    if token:
                        df_opt_chart = fetch_candle_data(api, token, "TEN_MINUTE", exchange="NFO", specific_date=chosen_date, days_back=60)
                        if not df_opt_chart.empty:
                            df_opt_chart = add_custom_ema(df_opt_chart)
                            df_opt_today_chart = df_opt_chart[df_opt_chart["datetime"].dt.date == chosen_date].copy()
                            lines = {"entry": t_data['entry_price'], "sl": t_data['SL'], "target": t_data['target']}
                            fig_opt = plot_strategy_with_trades(df_opt_today_chart, trades=[t_data], title=f"OPTION: {strike} {op_type} ({t_data['result']})", extra_lines=lines, is_option_chart=True)
                            st.plotly_chart(fig_opt, use_container_width=True)
                        else: st.error(f"No Data for {strike} {op_type}")
                    st.divider()

        if mode == "🔴 LIVE MARKET":
            analyzer_section.empty()
            
            # --- 1. PROCESS ALERTS & ORDERS ---
            for trade in trades_to_display:
                tid = trade["TradeID"]
                current_result = trade["result"]
                signal_time_str = trade["Signal Time"]
                last_known_state = st.session_state["trade_state"].get(tid, None)
                is_fresh_event = is_recent(signal_time_str, limit_minutes=15)

                if last_known_state is None:
                    if is_fresh_event:
                        st.toast(f"New Entry: #{tid}", icon="🚀")
                        
                        # --- PLACE ORDER LOGIC (DUAL) ---
                        angel_token = trade.get("token")
                        
                        if angel_token:
                            symbol_name = get_symbol_from_token(master_df, angel_token)
                            
                            if symbol_name:
                                # 1. Place Angel Order
                                angel_order_id = place_angel_order(api, angel_token, symbol_name, trade["entry_price"])
                                
                                # 2. Place Dhan Order
                                dhan_id = get_dhan_security_id(dhan_master_df, expiry_str, trade["Best Strike"], trade["type"])
                                dhan_order_id = "FAILED"
                                
                                if dhan_id:
                                    dhan_resp = place_dhan_order(dhan_api, dhan_id, trade["entry_price"])
                                    if dhan_resp:
                                        dhan_order_id = dhan_resp
                                        st.toast(f"Dhan Order Placed! ID: {dhan_resp}", icon="⚡")
                                else:
                                    st.error("Could not find Dhan Security ID for this strike")

                                # 3. Logging
                                if angel_order_id or dhan_order_id != "FAILED":
                                    combined_id = f"A:{angel_order_id} | D:{dhan_order_id}"
                                    st.toast(f"Orders Executed", icon="✅")
                                    
                                    order_details = {
                                        "Time": datetime.now(IST).strftime('%H:%M:%S'),
                                        "Symbol": symbol_name,
                                        "Type": "BUY (LIMIT)",
                                        "Qty": ORDER_QTY,
                                        "Price": trade["entry_price"],
                                        "Total Amount": ORDER_QTY * trade["entry_price"],
                                        "OrderID": combined_id
                                    }
                                    st.session_state["order_book"].append(order_details)
                                    send_email_notification(order_details, alert_type="ORDER")
                            else:
                                st.error("Symbol Name not found for Token!")
                        
                        # Send STRATEGY Email
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
            
            # --- 2. DISPLAY ORDER BOOK ---
            if st.session_state["order_book"]:
                st.write("### 📝 Order Book (Live Executions)")
                st.dataframe(pd.DataFrame(st.session_state["order_book"]), use_container_width=True)
                
    else:
        if st.session_state["session_trades"]:
            df_display = pd.DataFrame(st.session_state["session_trades"])
            styler = df_display[["TradeID", "Nature", "Signal Time", "entry_price", "SL", "target", "result", "Best Strike", "SL Time", "Target Time"]].style.map(lambda v: 'background-color: #006400' if 'TARGET' in str(v) else ('background-color: #8B0000' if 'SL' in str(v) else 'background-color: #00008B'), subset=['result'])
            main_table.dataframe(styler, use_container_width=True)
            main_table.info("⚠️ Showing last known data (Scan failed or returned empty).")
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
    if "stop_bot" not in st.session_state:
        st.session_state.stop_bot = False

    if st.sidebar.button("⏹ STOP BOT"):
        st.session_state.stop_bot = True
    
    if st.sidebar.button("▶️ RESTART"):
        st.session_state.stop_bot = False

    if not st.session_state.stop_bot:
        st.sidebar.success("✅ Bot Running...")
        
        now_time = datetime.now(IST).time()
        market_open = dtime(9, 15) <= now_time <= dtime(15, 30)
        
        if market_open:
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
            run_analysis_cycle()
            st.info("Market Closed. Bot is in Sleep Mode.")
    else:
        st.warning("Bot Manually Stopped.")
