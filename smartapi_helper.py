from SmartApi import SmartConnect
import pyotp
import pandas as pd
from config import API_KEY, CLIENT_CODE, PIN, TOTP_SECRET

def angel_login():
    totp = pyotp.TOTP(TOTP_SECRET).now()
    obj = SmartConnect(api_key=API_KEY)
    obj.generateSession(CLIENT_CODE, PIN, totp)
    return obj

def get_historical_data(obj, token, interval="TEN_MINUTE", date_str=None):
    from datetime import datetime
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": interval,
        "fromdate": f"{date_str} 09:15",
        "todate": f"{date_str} 15:30"
    }
    data = obj.getCandleData(params)
    if data['status'] and data['data']:
        df = pd.DataFrame(data['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    else:
        return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

