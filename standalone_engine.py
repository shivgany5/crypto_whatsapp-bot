import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
from scipy import stats
import traceback
from itertools import combinations

# Optional: Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ========================================== #
#               CONFIGURATION                #
# ========================================== #

# Real-Time Engine Config
SYMBOL = "BTCUSD"
RESOLUTION = "5m"
LOOKBACK_CANDLES = 500  # Fetch smaller window for efficiency
POLL_INTERVAL = 300     # 5 minutes in seconds (matches candle size)
DEFAULT_CSV = "data_standalone.csv"

# Trendline & Pivot Config
PIVOT_LEFT = 2
PIVOT_RIGHT = 2
MIN_R2 = 0.997
MIN_ABS_SLOPE = 1.5
MAX_POINT_RESIDUAL = 1e4
MIN_GAP = 4
MAX_GAP = 100
MIN_TOTAL_SPAN = 6
MAX_TOTAL_SPAN = 300
MIN_PRICE_RANGE = 0.0
MAX_LOOKAHEAD = 8
MAX_CLOSE_VIOL = 0.0
PADDING_CANDLES = 50

# Telegram Settings (Configured via Environment Variables)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# State Management to prevent duplicate alerts
SENT_ALERTS = set()

# ========================================== #
#               CORE LOGIC                   #
# ========================================== #

def fetch_latest_ohlcv_delta(symbol=SYMBOL, resolution=RESOLUTION, lookback_candles=LOOKBACK_CANDLES, save_path=DEFAULT_CSV):
    end_time = int(time.time())
    resolution_seconds = int(resolution.replace("m","")) * 60
    start_time = end_time - (lookback_candles * resolution_seconds)

    params = {
        "resolution": resolution,
        "symbol": symbol,
        "start": str(start_time),
        "end": str(end_time)
    }

    url = "https://api.india.delta.exchange/v2/history/candles"
    response = requests.get(url, params=params)
    data_json = response.json()

    if not data_json.get("success", False):
        raise Exception("Failed to fetch data from Delta Exchange")

    df = pd.DataFrame(data_json["result"])
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = df["timestamp"].astype(int)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df.to_csv(save_path, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetched {len(df)} candles from Delta Exchange")
    return df

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    required_cols = ["timestamp","open","high","low","close","volume"]
    df = df[required_cols]

    if df["timestamp"].astype(float).max() > 1e12:
        df["timestamp"] = (df["timestamp"].astype(float) / 1000).astype(int)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.reset_index().rename(columns={"index": "global_index"})

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["global_index", "timestamp", "datetime", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return df.reset_index(drop=True)

def find_pivot_highs(series: pd.Series, left: int, right: int) -> np.ndarray:
    n = len(series)
    highs = []
    for i in range(left, n - right):
        window = series[i - left : i + right + 1]
        val = series.iloc[i]
        if val == window.max() and (window == val).sum() == 1:
            highs.append(i)
    return np.array(highs, dtype=int)

def find_pivot_lows(series: pd.Series, left: int, right: int) -> np.ndarray:
    n = len(series)
    lows = []
    for i in range(left, n - right):
        window = series[i - left : i + right + 1]
        val = series.iloc[i]
        if val == window.min() and (window == val).sum() == 1:
            lows.append(i)
    return np.array(lows, dtype=int)

def fit_line(xs: np.ndarray, ys: np.ndarray):
    if len(xs) < 2:
        return 0.0, 0.0, 0.0
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    return float(slope), float(intercept), float(r_value ** 2)

def detect_three_point_trendlines(pivot_idx, wick_array, close_array, side):
    results = []
    pivot_idx = np.asarray(pivot_idx, dtype=int)
    n_pivots = len(pivot_idx)
    if n_pivots < 3:
        return results

    used_signatures = set()
    for base in range(n_pivots - 2):
        forward_range = range(base+1, min(n_pivots, base + MAX_LOOKAHEAD))
        for b, c in combinations(forward_range, 2):
            i1, i2, i3 = pivot_idx[base], pivot_idx[b], pivot_idx[c]
            if not (i1 < i2 < i3) or i3 >= len(wick_array):
                continue

            gap1, gap2, total_span = i2 - i1, i3 - i2, i3 - i1
            if not (MIN_GAP <= gap1 <= MAX_GAP) or not (MIN_GAP <= gap2 <= MAX_GAP) or not (MIN_TOTAL_SPAN <= total_span <= MAX_TOTAL_SPAN):
                continue

            y1, y2, y3 = wick_array[i1], wick_array[i2], wick_array[i3]
            ys = np.array([y1, y2, y3], dtype=float)
            xs = np.array([i1, i2, i3], dtype=float)
            xs_rel = xs - xs[0]

            slope, intercept, r2 = fit_line(xs_rel, ys)
            if r2 < MIN_R2 or abs(slope) < MIN_ABS_SLOPE:
                continue
    
            price_range = abs(y3 - y1)
            if price_range < MIN_PRICE_RANGE:
                continue

            y_hat = slope * xs_rel + intercept
            max_resid = np.max(np.abs(ys - y_hat))
            if max_resid > MAX_POINT_RESIDUAL:
                continue

            direction = "up" if slope > 0 else "down"

            if MAX_CLOSE_VIOL is not None and MAX_CLOSE_VIOL >= 0:
                seg_idx = np.arange(i1, i3 + 1, dtype=int)
                line_vals = slope * (seg_idx - i1) + intercept
                closes_seg = close_array[seg_idx]

                violation = False
                if side == "high" and direction == "down" and np.any(closes_seg > line_vals + MAX_CLOSE_VIOL):
                    violation = True
                elif side == "low" and direction == "up" and np.any(closes_seg < line_vals - MAX_CLOSE_VIOL):
                    violation = True
                if violation:
                    continue

            signature = (i1, i2, i3, round(slope, 6))
            if signature in used_signatures:
                continue
            used_signatures.add(signature)

            results.append({
                "side": side,
                "pivot_1_idx": int(i1), "pivot_2_idx": int(i2), "pivot_3_idx": int(i3),
                "gap_1": int(gap1), "gap_2": int(gap2), "total_span": int(total_span),
                "price_1": float(y1), "price_2": float(y2), "price_3": float(y3),
                "price_range": float(price_range),
                "slope": float(slope), "intercept": float(intercept),
                "r2": float(r2), "max_residual": float(max_resid),
                "direction": direction,
            })
    return results

# ========================================== #
#               ALERT NOTIFICATION           #
# ========================================== #

def plot_three_point_trendline_plotly(df: pd.DataFrame, pattern: dict, padding: int = PADDING_CANDLES):
    i1 = pattern["pivot_1_idx"]
    i2 = pattern["pivot_2_idx"]
    i3 = pattern["pivot_3_idx"]

    gstart = max(0, i1 - padding)
    gend   = min(len(df) - 1, i3 + padding)

    local_df = df.iloc[gstart:gend+1].copy()
    local_df.reset_index(drop=True, inplace=True)

    fig = go.Figure(data=[
        go.Candlestick(
            x=local_df["datetime"],
            open=local_df["open"],
            high=local_df["high"],
            low=local_df["low"],
            close=local_df["close"],
            name="Price",
            showlegend=False
        )
    ])

    slope = pattern["slope"]
    intercept = pattern["intercept"]

    avg_range = df["high"].sub(df["low"]).rolling(50).mean().iloc[-1]
    if pd.isna(avg_range) or avg_range == 0:
        avg_range = 1
    normalized_slope = slope / avg_range
    angle_deg = np.degrees(np.arctan(normalized_slope))

    global_idx = np.arange(gstart, gend + 1)
    xs_rel = global_idx - i1
    trend_y = slope * xs_rel + intercept

    fig.add_trace(go.Scatter(
        x=local_df["datetime"],
        y=trend_y,
        mode="lines",
        name="3-Point Trendline",
        line=dict(color="orange", width=2, dash="dash"),
        showlegend=False
    ))

    pivot_idxs = [
        pattern["pivot_1_idx"],
        pattern["pivot_2_idx"],
        pattern["pivot_3_idx"]
    ]

    pivot_x, pivot_y = [], []

    for pi in pivot_idxs:
        if gstart <= pi <= gend:
            local_i = pi - gstart
            pivot_x.append(local_df["datetime"].iloc[local_i])
            if pattern["side"] == "high":
                pivot_y.append(df["high"].iloc[pi])
            else:
                pivot_y.append(df["low"].iloc[pi])

    fig.add_trace(go.Scatter(
        x=pivot_x,
        y=pivot_y,
        mode="markers",
        marker=dict(size=8, color="deepskyblue"),
        name="Pivot Points",
        showlegend=False
    ))

    fig.update_layout(
        title=f"Pattern ({pattern['side'].upper()})<br>R²={pattern['r2']:.4f} | Slope={pattern['slope']:.4f} | Span={int(pattern['total_span'])} | Range={pattern['price_range']:.2f} | Angle={angle_deg:.1f}°",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        width=1000,
        margin=dict(l=30, r=30, b=30, t=60)
    )
    return fig

def send_telegram_alert(message_body, image_path=None):
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("Required Telegram credentials missing in ENV. Skipping Telegram alert.")
        return

    try:
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": message_body[:1024],
                "parse_mode": "HTML"
            }
            with open(image_path, "rb") as image_file:
                files = {"photo": image_file}
                response = requests.post(url, data=data, files=files)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message_body,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            
        if response.status_code == 200:
            print("🚀 Telegram alert sent successfully!")
        else:
            print(f"❌ Failed to send Telegram alert: {response.text}")
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")

def send_alert(pattern, df):
    detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = (
        f"🚨 <b>ALERT TRIGGERED AT:</b> {detection_time}\n\n"
        f"💎 <b>PATTERN:</b> {pattern['side'].upper()} Trendline ({pattern['direction']})\n"
        f"📉 <b>SYMBOL :</b> {SYMBOL} ({RESOLUTION})\n\n"
        f"1️⃣ <b>Pivot 1:</b> {pattern['datetime_1']} @ <b>{pattern['price_1']:.2f}</b>\n"
        f"2️⃣ <b>Pivot 2:</b> {pattern['datetime_2']} @ <b>{pattern['price_2']:.2f}</b>\n"
        f"3️⃣ <b>Pivot 3:</b> {pattern['datetime_3']} @ <b>{pattern['price_3']:.2f}</b> (Trigger)\n\n"
        f"📊 <b>METRICS:</b>\n"
        f"• Quality (R²) : {pattern['r2']:.4f}\n"
        f"• Slope : {pattern['slope']:.4f}\n"
        f"• Duration : {pattern['total_span']} candles\n"
        f"• Price Range : {pattern['price_range']:.2f}\n"
    )
    print(f"\n{'='*60}\n{message.replace('<b>', '').replace('</b>', '')}{'='*60}\n")  
    
    os.makedirs("charts", exist_ok=True)
    image_filename = f"charts/pattern_{int(time.time())}_{pattern['side']}.png"
    try:
        fig = plot_three_point_trendline_plotly(df, pattern)
        fig.write_image(image_filename)
        send_telegram_alert(message, image_path=image_filename)
    except Exception as e:
        print(f"Failed to generate plot: {e}")
        send_telegram_alert(message)

# ========================================== #
#               MAIN LOOP ENGINE             #
# ========================================== #

def process_and_scan():
    try:
        # 1. Fetch Latest Data
        fetch_latest_ohlcv_delta(symbol=SYMBOL, resolution=RESOLUTION, lookback_candles=LOOKBACK_CANDLES, save_path=DEFAULT_CSV)

        # 2. Load Data
        df = load_csv(DEFAULT_CSV)
        if df.empty:
            print("Warning: Empty dataframe fetched.")
            return

        # 3. Detect Pivots
        high_series, low_series = df["high"], df["low"]
        high_pivots = find_pivot_highs(high_series, PIVOT_LEFT, PIVOT_RIGHT)
        low_pivots = find_pivot_lows(low_series, PIVOT_LEFT, PIVOT_RIGHT)

        high_array, low_array, close_array = high_series.to_numpy(dtype=float), low_series.to_numpy(dtype=float), df["close"].to_numpy(dtype=float)

        # 4. Detect Trendlines
        high_trends = detect_three_point_trendlines(high_pivots, high_array, close_array, "high")
        low_trends = detect_three_point_trendlines(low_pivots, low_array, close_array, "low")

        all_trends = high_trends + low_trends

        # 5. Filter for High Quality
        filtered_trends = [
            t for t in all_trends
            if (t["side"] == "high" and t["direction"] == "down")
            or (t["side"] == "low" and t["direction"] == "up")
        ]

        # 6. Check for New Alerts
        new_alerts_count = 0
        for t in filtered_trends:
            i1, i2, i3 = t["pivot_1_idx"], t["pivot_2_idx"], t["pivot_3_idx"]
            ts1, ts2, ts3 = df.loc[i1, "timestamp"], df.loc[i2, "timestamp"], df.loc[i3, "timestamp"]
            pattern_id = (ts1, ts2, ts3)

            if pattern_id not in SENT_ALERTS:
                t["datetime_1"], t["datetime_2"], t["datetime_3"] = df.loc[i1, "datetime"], df.loc[i2, "datetime"], df.loc[i3, "datetime"]
                
                target_col = t["side"] 
                t["price_1"], t["price_2"], t["price_3"] = df.loc[i1, target_col], df.loc[i2, target_col], df.loc[i3, target_col]
                
                send_alert(t, df)
                SENT_ALERTS.add(pattern_id)
                new_alerts_count += 1
        
        if new_alerts_count == 0:
            print(f"Scan complete. No new patterns. (Total tracked: {len(SENT_ALERTS)})")

    except Exception as e:
        print(f"Error in scan loop: {e}")
        traceback.print_exc()

def main():
    print(f"Starting Standalone Real-Time Engine for {SYMBOL} ({RESOLUTION})")
    print("Press Ctrl+C to stop.")
    
    while True:
        process_and_scan()
        print(f"Sleeping for {POLL_INTERVAL} seconds...")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
