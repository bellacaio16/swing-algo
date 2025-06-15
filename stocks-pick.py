import os
import datetime
import pandas as pd
import numpy as np
import time
import logging
import pyotp
from scipy.signal import argrelextrema
from SmartApi import SmartConnect, smartExceptions
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ======= CONFIGURATION ========
SYMBOL_TOKEN_FILE = 'symbol-token.txt'
HISTORY_YEARS = 2
ANALYSIS_DAYS = 60                # days to analyze
OUTPUT_CSV = 'breakouts.csv'
CACHE_DIR = 'cache'
HOURLY_INTERVAL = 'ONE_HOUR'
DAILY_INTERVAL = 'ONE_DAY'
FIVEMIN_INTERVAL = 'FIVE_MINUTE'
TENMIN_INTERVAL = 'TEN_MINUTE'
MAX_WORKERS = 5                  # parallel symbol threads
API_RATE_LIMIT = 5               # max API calls per second

# Breakout criteria
MIN_LEVEL_AGE = 30                # days
MIN_TOUCHES = 2
VOL_MULTIPLIER = 1.5
RSI_RANGE = {'bullish': (55, 75), 'bearish': (25, 40)}
CONFIRMATION_CANDLES = 3          # Candles to confirm breakout and MACD crossover

# Technical parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOL_AVG_WINDOW = 20
PIVOT_WINDOW_HOURLY = 24          # periods

# Risk management
SL_BUFFER = 0.03   # 3%
TARGET1_PCT = 0.08
TARGET2_PCT = 0.15

# SmartAPI credentials
API_KEY = '3ZkochvK'
USERNAME = 'D61366376'
PASSWORD = '2299'
TOTP_SECRET = 'B4C2S5V6DUWUP2E4SFVRWA5CGE'
# ==============================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_api_lock = Lock()
_last_api_time = 0.0

def rate_limited_call(fn, *args, **kwargs):
    global _last_api_time
    with _api_lock:
        elapsed = time.time() - _last_api_time
        wait = max(0, (1.0 / API_RATE_LIMIT) - elapsed)
        if wait > 0:
            time.sleep(wait)
        result = fn(*args, **kwargs)
        _last_api_time = time.time()
        return result


def load_symbol_tokens(path):
    df = pd.read_csv(path, names=['symbol','token'], dtype=str)
    return {row.symbol: row.token for _, row in df.iterrows() if row.token and row.token != 'NOT_FOUND'}


def init_smartapi():
    smart = SmartConnect(API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    sess = rate_limited_call(smart.generateSession, USERNAME, PASSWORD, totp)
    if not sess.get('status'):
        raise RuntimeError('SmartAPI login failed')
    rate_limited_call(smart.generateToken, sess['data']['refreshToken'])
    return smart


def fetch_candles(smart, symbol, token, start_date, end_date, interval):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = f"{token}_{interval}.pkl"
    cache_file = os.path.join(CACHE_DIR, cache_key)

    # Load existing cache
    if os.path.exists(cache_file):
        try:
            df_all = pd.read_pickle(cache_file)
        except:
            df_all = pd.DataFrame()
    else:
        df_all = pd.DataFrame()

    # Determine fetch start
    if not df_all.empty:
        last_date = df_all.index.max().date()
        fetch_start = max(last_date + datetime.timedelta(days=1), start_date)
    else:
        fetch_start = start_date

    # Fetch only new data
    if fetch_start <= end_date:
        params = {
            'exchange': 'NSE',
            'symboltoken': token,
            'interval': interval,
            'fromdate': fetch_start.strftime('%Y-%m-%d 09:15'),
            'todate':   end_date.strftime('%Y-%m-%d 15:30')
        }
        try:
            resp = rate_limited_call(smart.getCandleData, params)
            data = resp.get('data') or []
            if data:
                df_new = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
                df_new.set_index('timestamp', inplace=True)
                # Combine and dedupe
                df_all = pd.concat([df_all, df_new])
                df_all = df_all[~df_all.index.duplicated(keep='last')].sort_index()
                # Trim to window
                df_all = df_all[df_all.index.date >= start_date]
                df_all.to_pickle(cache_file)
        except smartExceptions.DataException:
            pass
        except Exception as e:
            logger.warning(f"[{symbol}] Fetch error: {e}")
    return df_all

# compute_rsi, compute_macd, find_pivots, calculate_level_strength, find_strong_levels, round_target, analyze_stock remain unchanged





def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line

def find_pivots(high, low, window=5):
    max_idx = argrelextrema(high.values, np.greater, order=window)[0]
    min_idx = argrelextrema(low.values, np.less, order=window)[0]
    return high.iloc[max_idx], low.iloc[min_idx]

def calculate_level_strength(df, level, typ, tol=0.015):
    touches = 0
    first_t = None
    for idx, row in df.iterrows():
        price = row['high'] if typ=='resistance' else row['low']
        if abs(price - level) / level <= tol:
            touches += 1
            first_t = first_t or idx
    if touches < MIN_TOUCHES:
        return 0, None
    age = (df.index.max().date() - first_t.date()).days
    score = min(touches * 0.3 + min(age/30, 2) * 0.7, 1.0)
    return score, age

def find_strong_levels(df, pivot_window=5):
    recent = df if len(df) <= VOL_AVG_WINDOW*6 else df.iloc[-VOL_AVG_WINDOW*6:]
    piv_res, piv_sup = find_pivots(recent['high'], recent['low'], window=pivot_window)
    levels = {'res': [], 'sup': []}
    current = df['close'].iloc[-1]
    for lvl in np.unique(piv_res):
        score, age = calculate_level_strength(df, lvl, 'resistance')
        if score >= 0.4 and lvl > current and age >= MIN_LEVEL_AGE:
            levels['res'].append({'level': lvl, 'score': score, 'age': age})
    for lvl in np.unique(piv_sup):
        score, age = calculate_level_strength(df, lvl, 'support')
        if score >= 0.4 and lvl < current and age >= MIN_LEVEL_AGE:
            levels['sup'].append({'level': lvl, 'score': score, 'age': age})
    levels['res'].sort(key=lambda x: x['level'])
    levels['sup'].sort(key=lambda x: x['level'], reverse=True)
    return levels

def round_target(price):
    if price < 100:
        return round(price, 2)
    elif price < 500:
        return round(price, 1)
    else:
        return round(price)

def analyze_stock(smart, symbol, token):
    logger.info(f"Analyzing {symbol}")
    today = datetime.date.today()
    start_hist = today - datetime.timedelta(days=HISTORY_YEARS*365)
    start_intra = today - datetime.timedelta(days=ANALYSIS_DAYS)

    # 1) Pivot data
    df_hist = fetch_candles(smart, symbol, token, start_hist, today, HOURLY_INTERVAL)
    if df_hist.empty:
        df_hist = fetch_candles(smart, symbol, token, start_hist, today, DAILY_INTERVAL)
    if df_hist.empty or len(df_hist) < 50:
        return []

    # 2) Intraday bulk
    df_intra = fetch_candles(smart, symbol, token, start_intra, today, FIVEMIN_INTERVAL)
    if df_intra.empty:
        df_intra = fetch_candles(smart, symbol, token, start_intra, today, TENMIN_INTERVAL)
    if df_intra.empty:
        return []

    # compute indicators
    df_intra['rsi'] = compute_rsi(df_intra['close'])
    df_intra['macd_line'], df_intra['signal_line'] = compute_macd(df_intra['close'])
    df_intra['macd_hist'] = df_intra['macd_line'] - df_intra['signal_line']
    df_intra['vol_avg'] = df_intra['volume'].rolling(VOL_AVG_WINDOW).mean()

    results = []
    for i in range(1, ANALYSIS_DAYS+1):
        day = today - datetime.timedelta(days=i)
        piv_df = df_hist[df_hist.index.date < day]
        if len(piv_df) < 50:
            continue
        levels = find_strong_levels(piv_df, pivot_window=PIVOT_WINDOW_HOURLY)
        if not levels['res'] and not levels['sup']:
            continue

        day_df = df_intra[df_intra.index.date == day]
        if day_df.empty:
            continue

        for ts, row in day_df.iterrows():
            # position index for MACD
            try:
                pos = df_intra.index.get_loc(ts)
            except KeyError:
                continue
            if pos < CONFIRMATION_CANDLES:
                continue
            # skip if insufficient data
            if np.isnan(row['vol_avg']) or np.isnan(row['rsi']):
                continue

            # MACD crossover logic
            hist = df_intra['macd_hist']
            macd_bullish      = (hist.iloc[pos] > 0 and hist.iloc[pos-1] <= 0)
            macd_stable       = all(hist.iloc[pos-offset] > 0 for offset in range(CONFIRMATION_CANDLES))
            macd_bearish      = (hist.iloc[pos] < 0 and hist.iloc[pos-1] >= 0)
            macd_stable_bear  = all(hist.iloc[pos-offset] < 0 for offset in range(CONFIRMATION_CANDLES))

            action, lvl = None, None
            # BUY check
            if levels['res'] and row['volume'] > VOL_MULTIPLIER * row['vol_avg'] \
               and RSI_RANGE['bullish'][0] <= row['rsi'] <= RSI_RANGE['bullish'][1] \
               and macd_bullish and macd_stable:
                action = 'BUY'
                res_lvls = [l['level'] for l in levels['res'] if l['level'] < row['close']]
                lvl = {'level': max(res_lvls), 'score': None, 'age': None} if res_lvls else None
            # SELL check
            if not action and levels['sup'] and row['volume'] > VOL_MULTIPLIER * row['vol_avg'] \
               and RSI_RANGE['bearish'][0] <= row['rsi'] <= RSI_RANGE['bearish'][1] \
               and macd_bearish and macd_stable_bear:
                action = 'SELL'
                sup_lvls = [l['level'] for l in levels['sup'] if l['level'] > row['close']]
                lvl = {'level': min(sup_lvls), 'score': None, 'age': None} if sup_lvls else None

            if not action or not lvl:
                continue

            bp = round_target(lvl['level'])
            sl_price = bp * (1 - SL_BUFFER) if action == 'BUY' else bp * (1 + SL_BUFFER)

            # compute targets
            piv_list = levels['res'] if action == 'BUY' else levels['sup']
            pts = [l['level'] for l in piv_list if (l['level'] > bp if action == 'BUY' else l['level'] < bp)]
            t_1 = pts[0] if len(pts) > 0 else bp * (1 + (TARGET1_PCT if action == 'BUY' else -TARGET1_PCT))
            t_2 = pts[1] if len(pts) > 1 else bp * (1 + (TARGET2_PCT if action == 'BUY' else -TARGET2_PCT))

            if action=='BUY':
                sorted_targets = sorted([bp * 1.06, t_1, t_2])
            else:
                sorted_targets = sorted([bp * (1-0.06), t_1, t_2], reverse=True)

            t1, t2, t3 = map(round_target, sorted_targets)

            results.append({
                'date': day.strftime('%Y-%m-%d'),
                'time': ts.strftime('%H:%M'),
                'symbol': symbol,
                'token': token,
                'action': action,
                'breakout_level': bp,
                'stop_loss': sl_price,
                'target1': t1,
                'target2': t2,
                'target3': t3,
                'rsi': row['rsi'],
                'macd_hist': hist.iloc[pos],
                'volume': row['volume'],
                'level_score': lvl.get('score'),
                'level_age': lvl.get('age'),
                'close': row['close']
            })
            break

    return results


# ... other function definitions unchanged ...

def main():
    smart = init_smartapi()
    symbols = load_symbol_tokens(SYMBOL_TOKEN_FILE)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_stock, smart, sym, tok): sym for sym, tok in symbols.items()}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                picks = future.result()
                if isinstance(picks, list):
                    results.extend(picks)
            except Exception as e:
                logger.error(f"{sym} analysis failed: {e}")

    # Consolidate and save
    df = pd.DataFrame(results)
    if df.empty or 'action' not in df.columns:
        logger.warning("No valid picks returned; writing empty CSV.")
        pd.DataFrame().to_csv(OUTPUT_CSV, index=False)
        return

    df = df[df['action'] == 'BUY']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.time
    df = df.dropna(subset=['date', 'time'])
    df = df.sort_values(['symbol','date','time'])
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == '__main__':
    main()


