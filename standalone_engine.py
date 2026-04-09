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
from typing import Dict, Any, List, Optional, Tuple

# Optional: Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure Kaleido for Railway deployment
# Set Chromium path for Kaleido using environment variable (recommended approach)
chromium_path = os.getenv('CHROMIUM_PATH', '/usr/bin/chromium')
if os.path.exists(chromium_path):
    os.environ['KALEIDO_CHROMIUM_PATH'] = chromium_path
    print(f"[INFO] Kaleido configured to use Chromium at: {chromium_path}")
else:
    print(f"[WARNING] Chromium not found at {chromium_path}, image export may fail")

# ========================================== #
#               CONFIGURATION                #
# ========================================== #

# Real-Time Engine Config
SYMBOL = "BTCUSD"
RESOLUTION = "5m"
LOOKBACK_CANDLES = 250 # was 500
POLL_INTERVAL = 300     # 5 minutes in seconds (matches candle size)
DEFAULT_CSV = "data_standalone.csv"

# Trendline & Pivot Config
PIVOT_LEFT = 2
PIVOT_RIGHT = 2
MIN_R2 = 0.996
MIN_ABS_SLOPE = 1.5
MAX_ABS_SLOPE = 16.5           # Upper slope cap
MAX_POINT_RESIDUAL = 1e4
MIN_GAP = 6                    # was 4
MAX_GAP = 100
MIN_TOTAL_SPAN = 6
MAX_TOTAL_SPAN = 300
MIN_PRICE_RANGE = 0.0
MAX_LOOKAHEAD = 13             # was 8
MAX_CLOSE_VIOL = 0.0
PADDING_CANDLES = 50           # was 50

# --- Pattern Rejection Thresholds ---
MIN_DIST_RATIO    = 0.65       # Compression filter threshold
MIN_MAX_GAP_RATIO = 0.65       # Compression filter threshold

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
            if r2 < MIN_R2 or abs(slope) < MIN_ABS_SLOPE or abs(slope) > MAX_ABS_SLOPE:
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
#   COUNTER PIVOT AUDIT  (info-generator)    #
# ========================================== #

def anchor_line_value(tl: Dict, idx: int) -> float:
    """Value of the anchor trendline at a global candle index."""
    return tl["slope"] * (idx - tl["pivot_1_idx"]) + tl["intercept"]


# How many candles before pivot_1 to start the sweep
COUNTER_LOOKBACK_PRE = 5


def counter_pivot_audit(tl: Dict, df: pd.DataFrame,
                        all_pivot_highs: np.ndarray,
                        all_pivot_lows: np.ndarray) -> Dict:
    """
    Pure info-generator — sweeps EVERY candle from (pivot_1 - 5) to pivot_3.

    For anchor = support  ("low"):      opposing price = candle HIGH
    For anchor = resistance ("high"):   opposing price = candle LOW

    Builds a per-candle gap series (distance of opposing price from anchor
    trendline) and evaluates compression through 4 independent lenses.

    is_compressing = True only when ALL 4 lenses confirm.
    Returns a dict of all metrics. Does NOT filter or reject anything.
    """
    i1   = tl["pivot_1_idx"]
    i3   = tl["pivot_3_idx"]
    side = tl["side"]

    n        = len(df)
    high_arr = df["high"].to_numpy(dtype=float)
    low_arr  = df["low"].to_numpy(dtype=float)

    # ── Sweep window ──────────────────────────────────────────────────────
    sweep_start = max(0, i1 - COUNTER_LOOKBACK_PRE)
    sweep_end   = min(n - 1, i3)
    sweep_idx   = np.arange(sweep_start, sweep_end + 1, dtype=int)
    n_sweep     = len(sweep_idx)

    if side == "low":
        counter_prices = high_arr[sweep_idx]   # support anchor → opposing = highs
    else:
        counter_prices = low_arr[sweep_idx]    # resistance anchor → opposing = lows

    anchor_vals = np.array([anchor_line_value(tl, int(ci)) for ci in sweep_idx])

    # Gap series: |opposing_price - anchor_line| at every candle in window
    gaps = np.abs(counter_prices - anchor_vals)

    # Split exactly in half for ratio-based metrics
    half = max(1, n_sweep // 2)
    gaps_h1 = gaps[:half]
    gaps_h2 = gaps[half:]

    # ── METRIC 1 : Distance Sum Ratio ────────────────────────────────────
    sum_h1     = float(np.sum(gaps_h1))
    sum_h2     = float(np.sum(gaps_h2))
    dist_ratio = sum_h1 / sum_h2 if sum_h2 > 0 else 1.0

    # ── METRIC 2 : Max Gap Ratio ──────────────────────────────────────────
    max_h1        = float(np.max(gaps_h1)) if len(gaps_h1) else 0.0
    max_h2        = float(np.max(gaps_h2)) if len(gaps_h2) else 0.0
    max_gap_ratio = max_h1 / max_h2 if max_h2 > 0 else 1.0

    max_h1_idx = int(sweep_idx[int(np.argmax(gaps_h1))]) if len(gaps_h1) else int(sweep_idx[0])
    max_h2_idx = int(sweep_idx[half + int(np.argmax(gaps_h2))]) if len(gaps_h2) else int(sweep_idx[-1])

    # ── METRIC 3 : Gap Slope (info only — not used in score) ─────────────
    gap_slope_raw, _, _ = fit_line(np.arange(n_sweep, dtype=float), gaps)
    mean_gap             = float(np.mean(gaps)) if np.mean(gaps) > 0 else 1.0
    gap_slope_norm       = gap_slope_raw / mean_gap

    # ── METRIC 4 : Consistency Score (info only — not used in score) ──────
    if len(gaps) >= 2:
        consistency_score = float(np.sum(np.diff(gaps) < 0) / (len(gaps) - 1))
    else:
        consistency_score = 0.0

    # ── Quad-confirmation compression flag ───────────────────────────────
    is_compressing = (
        dist_ratio        > 1.0
        and max_gap_ratio > 1.0
        and gap_slope_norm < 0.0
        and consistency_score > 0.5
    )

    # ── Energy state (driven by dist_ratio) ──────────────────────────────
    if dist_ratio >= 1.5:
        energy_state = "HIGH_COMPRESSION"
    elif dist_ratio >= 1.0:
        energy_state = "MILD_COMPRESSION"
    elif dist_ratio >= 0.8:
        energy_state = "PARALLEL"
    else:
        energy_state = "EXPANSION"

    # ── N-period tightest-point check ────────────────────────────────────
    RANGE_LOOKBACK   = 20
    RANGE_PERCENTILE = 35
    lb_start      = max(0, i3 - RANGE_LOOKBACK)
    lb_ranges     = high_arr[lb_start:i3 + 1] - low_arr[lb_start:i3 + 1]
    threshold     = np.percentile(lb_ranges, RANGE_PERCENTILE)
    current_range = high_arr[i3] - low_arr[i3]
    at_tightest   = bool(current_range <= threshold)

    # ── COMPRESSION SCORE  (0.0 → 1.0) ──────────────────────────────────
    TL_SLOPE_MAX = 50.0

    n_dist      = float(np.clip((dist_ratio    - 1.0) / 2.0,                       0.0, 1.0))
    n_maxgap    = float(np.clip((max_gap_ratio - 1.0) / 2.0,                       0.0, 1.0))
    tl_slope_abs = abs(tl["slope"])
    n_tl_slope  = float(np.clip((tl_slope_abs - MIN_ABS_SLOPE) / (TL_SLOPE_MAX - MIN_ABS_SLOPE), 0.0, 1.0))

    compression_score = (
        0.4 * n_dist
      + 0.4 * n_maxgap
      + 0.2 * n_tl_slope
    )

    # Grade
    if compression_score >= 0.70:
        compression_grade = "STRONG"
    elif compression_score >= 0.45:
        compression_grade = "MEDIUM"
    else:
        compression_grade = "WEAK"

    return dict(
        sweep_idx          = sweep_idx.tolist(),
        counter_prices     = counter_prices.tolist(),
        anchor_vals        = anchor_vals.tolist(),
        gaps               = gaps.tolist(),
        sum_h1             = sum_h1,
        sum_h2             = sum_h2,
        dist_ratio         = dist_ratio,
        max_h1             = max_h1,
        max_h2             = max_h2,
        max_gap_ratio      = max_gap_ratio,
        max_h1_idx         = max_h1_idx,
        max_h2_idx         = max_h2_idx,
        gap_slope_norm     = float(gap_slope_norm),
        consistency_score  = consistency_score,
        n_dist             = n_dist,
        n_maxgap           = n_maxgap,
        n_tl_slope         = n_tl_slope,
        tl_slope_abs       = tl_slope_abs,
        compression_score  = float(compression_score),
        compression_grade  = compression_grade,
        is_compressing     = bool(is_compressing),
        energy_state       = energy_state,
        at_tightest        = at_tightest,
        n_candles_swept    = int(n_sweep),
    )


def enrich_trendlines_with_compression(trendlines: List[Dict],
                                       df: pd.DataFrame,
                                       all_pivot_highs: np.ndarray,
                                       all_pivot_lows: np.ndarray) -> List[Dict]:
    """
    Runs counter_pivot_audit on every trendline and attaches the result
    under the key 'compression' inside the trendline dict.

    No filtering. Returns the same list with extra metadata.
    """
    for tl in trendlines:
        tl["compression"] = counter_pivot_audit(
            tl, df, all_pivot_highs, all_pivot_lows
        )
    return trendlines


def filter_compressing_trendlines(trendlines: List[Dict]) -> List[Dict]:
    """
    Post-enrichment filter — keeps only trendlines where
    counter_pivot_audit confirmed compression (is_compressing = True).

    Call this AFTER enrich_trendlines_with_compression.
    Does not modify the original list.
    """
    return [tl for tl in trendlines
            if tl.get("compression", {}).get("is_compressing", False)]

# ========================================== #
#               ALERT NOTIFICATION           #
# ========================================== #

def plot_three_point_trendline_plotly(df: pd.DataFrame,
                                       pattern: Dict[str, Any],
                                       padding: int = PADDING_CANDLES):
    """
    Candlestick + anchor trendline + compression overlays.

    Visual layers:
      • Candlestick chart
      • Anchor trendline (lime = support, red = resistance)
      • Yellow circles   : 3 anchor pivot points
      • Filled area      : between anchor line and opposing candle prices
                           (the "gap zone" — shows the coil visually)
      • Counter line     : continuous line of opposing candle highs/lows
                           across the full sweep window (i1-5 → i3)
      • Title            : R², Slope, Angle + CR, GapShrink, energy state
    """
    i1   = pattern["pivot_1_idx"]
    i2   = pattern["pivot_2_idx"]
    i3   = pattern["pivot_3_idx"]
    side = pattern["side"]

    gstart = max(0, i1 - padding)
    gend   = min(len(df) - 1, i3 + padding)

    local_df = df.iloc[gstart:gend + 1].copy()
    local_df.reset_index(drop=True, inplace=True)

    # ── Candlestick ───────────────────────────────────────────────────────
    fig = go.Figure(data=[go.Candlestick(
        x=local_df["datetime"],
        open=local_df["open"], high=local_df["high"],
        low=local_df["low"],   close=local_df["close"],
        name="Price", showlegend=False
    )])

    # ── Anchor trendline ──────────────────────────────────────────────────
    slope     = pattern["slope"]
    intercept = pattern["intercept"]

    global_idx = np.arange(gstart, gend + 1)
    xs_rel     = global_idx - i1
    trend_y    = slope * xs_rel + intercept

    tl_color = "lime" if side == "low" else "red"
    fig.add_trace(go.Scatter(
        x=local_df["datetime"], y=trend_y,
        mode="lines", name="Anchor Trendline",
        line=dict(color=tl_color, width=2, dash="dash"),
        showlegend=True
    ))

    # ── 3 Anchor pivot markers (yellow circles) ───────────────────────────
    price_arr = df["high"].to_numpy() if side == "high" else df["low"].to_numpy()
    for pidx in [i1, i2, i3]:
        if gstart <= pidx <= gend:
            fig.add_trace(go.Scatter(
                x=[df["datetime"].iloc[pidx]],
                y=[price_arr[pidx]],
                mode="markers",
                marker=dict(size=11, color="yellow", symbol="circle",
                            line=dict(color="black", width=1)),
                name="Anchor Pivot" if pidx == i1 else None,
                showlegend=(pidx == i1)
            ))

    # ── Compression overlays ──────────────────────────────────────────────
    comp = pattern.get("compression")
    if comp:
        sweep_idx      = np.array(comp["sweep_idx"])
        counter_prices = np.array(comp["counter_prices"])
        anchor_vals    = np.array(comp["anchor_vals"])

        # Restrict to visible plot window
        mask        = (sweep_idx >= gstart) & (sweep_idx <= gend)
        vis_idx     = sweep_idx[mask]
        vis_counter = counter_prices[mask]
        vis_anchor  = anchor_vals[mask]

        c_line_color = "orange"                   if side == "low" else "deepskyblue"
        dot_color    = "rgba(255,165,0,0.50)"     if side == "low" else "rgba(0,200,255,0.50)"
        fill_color   = "rgba(255,165,0,0.10)"     if side == "low" else "rgba(0,200,255,0.10)"

        if len(vis_idx) > 0:
            vis_dt = df["datetime"].iloc[vis_idx].values

            # ── Counter price line ────────────────────────────────────────
            fig.add_trace(go.Scatter(
                x=vis_dt, y=vis_counter,
                mode="lines",
                name="Counter price",
                line=dict(color=c_line_color, width=1.2),
                showlegend=True
            ))

            # ── Filled gap zone ───────────────────────────────────────────
            fig.add_trace(go.Scatter(
                x=np.concatenate([vis_dt, vis_dt[::-1]]),
                y=np.concatenate([vis_counter, vis_anchor[::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(0,0,0,0)"),
                name="Gap Zone",
                showlegend=True,
                hoverinfo="skip"
            ))

            # ── Counter dot markers at every sweep candle ─────────────────
            fig.add_trace(go.Scatter(
                x=vis_dt, y=vis_counter,
                mode="markers",
                marker=dict(size=4, color=dot_color, symbol="circle"),
                name="Counter candles",
                showlegend=True
            ))

            # ── Gap lines: anchor → counter at every sweep candle ─────────
            gap_x, gap_y = [], []
            for ci, cp, av in zip(vis_idx, vis_counter, vis_anchor):
                dt_val = df["datetime"].iloc[int(ci)]
                gap_x += [dt_val, dt_val, None]
                gap_y += [float(av), float(cp), None]

            fig.add_trace(go.Scatter(
                x=gap_x, y=gap_y,
                mode="lines",
                name="Gap lines",
                line=dict(color="rgba(180,180,180,0.22)", width=1, dash="dot"),
                showlegend=True
            ))

        # ── Max gap highlights ────────────────────────────────────────────
        for mkey, mlabel, mcolor in [
            ("max_h1_idx", "Max Gap H1 (1st half)", "magenta"),
            ("max_h2_idx", "Max Gap H2 (2nd half)", "white"),
        ]:
            midx = comp.get(mkey)
            if midx is not None and gstart <= midx <= gend:
                pos = np.where(sweep_idx == midx)[0]
                if len(pos):
                    cp  = float(counter_prices[pos[0]])
                    av  = float(anchor_vals[pos[0]])
                    mdt = df["datetime"].iloc[int(midx)]

                    # Star marker at the counter price
                    fig.add_trace(go.Scatter(
                        x=[mdt], y=[cp],
                        mode="markers",
                        marker=dict(size=14, color=mcolor, symbol="star",
                                    line=dict(color="black", width=1)),
                        name=mlabel,
                        showlegend=True
                    ))
                    # Bold gap line for this max-gap candle
                    fig.add_trace(go.Scatter(
                        x=[mdt, mdt], y=[av, cp],
                        mode="lines",
                        line=dict(color=mcolor, width=2.5),
                        showlegend=False
                    ))

        # ── Title ─────────────────────────────────────────────────────────
        score = comp["compression_score"]
        grade = comp["compression_grade"]
        dr    = comp["dist_ratio"]
        mgr   = comp["max_gap_ratio"]
        state = comp["energy_state"]
        flag  = "✅ COMPRESSING" if comp["is_compressing"] else "❌ not compressing"
        tight = " 🎯" if comp["at_tightest"] else ""
        comp_str = (f" | Score={score:.2f} [{grade}]  {flag}{tight}<br>"
                    f"DistR={dr:.2f}  MaxGapR={mgr:.2f}  [{state}]")
    else:
        comp_str = ""

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=(f"Pattern ({side.upper()})"
               f" | R²={pattern['r2']:.4f}  Slope={slope:.2f}{comp_str}"),
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=700,
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
    comp = pattern.get("compression", {})
    message += (
        f"• Compression Score : {comp.get('compression_score', 0.0):.2f} [{comp.get('compression_grade', '-')}]\n"
        f"• Energy State      : {comp.get('energy_state', '-')}\n"
        f"• Dist Ratio        : {comp.get('dist_ratio', 0.0):.2f}\n"
        f"• MaxGap Ratio      : {comp.get('max_gap_ratio', 0.0):.2f}\n"
        f"• Compressing       : {'✅ YES' if comp.get('is_compressing') else '❌ NO'}\n"
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

        # Enrich all trendlines with compression metadata
        enrich_trendlines_with_compression(all_trends, df, high_pivots, low_pivots)

        # 5. Filter for High Quality
        filtered_trends = [
            t for t in all_trends
            if (t["side"] == "high" and t["direction"] == "down")
            or (t["side"] == "low" and t["direction"] == "up")
        ]

        # Compression threshold filter
        filtered_trends = [
            t for t in filtered_trends
            if t.get("compression", {}).get("dist_ratio", 0.0) >= MIN_DIST_RATIO
            and t.get("compression", {}).get("max_gap_ratio", 0.0) >= MIN_MAX_GAP_RATIO
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
