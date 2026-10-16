import os
import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Reads your credentials securely from GitHub Secrets
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
UNIVERSE_MODE = os.getenv("UNIVERSE_MODE", "NIFTY500")

def get_nse_tickers(mode="NIFTY500"):
    """Fetches official NSE stock symbols directly from NSE India archives."""
    headers = {"User-Agent": "Mozilla/5.0"}
    if mode == "NIFTY500":
        url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_csv(io.StringIO(res.text))
        symbols = df["Symbol"].dropna().unique().tolist()
    else:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_csv(io.StringIO(res.text))
        df.columns = [col.strip() for col in df.columns]
        df = df[df["SERIES"] == "EQ"]
        symbols = df["SYMBOL"].dropna().unique().tolist()
    return [f"{sym}.NS" for sym in symbols]

def calculate_rsi(series, period=14):
    """Calculates Weekly Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def scan_stock(ticker):
    """Downloads weekly data and checks all 3 institutional breakout strategies."""
    try:
        df = yf.download(ticker, period="5y", interval="1wk", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Require at least 205 weeks of trading history
        if df.empty or len(df) < 205:
            return None

        # Core Moving Averages & Volume
        df["WMA_200"] = df["Close"].rolling(window=200).mean()
        df["WMA_30"] = df["Close"].rolling(window=30).mean()
        df["Vol_Avg_20"] = df["Volume"].rolling(window=20).mean()

        # Prior 20-Week and 52-Week Highs (shifted by 1 week to exclude current candle)
        df["High_20W"] = df["High"].shift(1).rolling(window=20).max()
        df["High_52W"] = df["High"].shift(1).rolling(window=52).max()
        df["RSI_14"] = calculate_rsi(df["Close"], period=14)

        # Prior 4-Week Darvas Box Width (measures consolidation tightness before breakout)
        prior_4w_high = df["High"].shift(1).rolling(window=4).max()
        prior_4w_low = df["Low"].shift(1).rolling(window=4).min()
        df["Base_Width_4W"] = (prior_4w_high - prior_4w_low) / prior_4w_low

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Gate 1: Liquidity Filter (Price >= Rs 50 and 20W Avg Volume >= 1,00,000)
        if curr["Close"] < 50 or curr["Vol_Avg_20"] < 100000:
            return None

        # Gate 2: Weekly Candle Closing Strength (Top 35% of weekly high-low range)
        weekly_range = curr["High"] - curr["Low"]
        if weekly_range <= 0:
            return None
        close_position = (curr["Close"] - curr["Low"]) / weekly_range
        strong_candle = close_position >= 0.65

        # Gate 3: Institutional Volume Expansion (>= 1.5x of 20-week average)
        vol_ratio = curr["Volume"] / curr["Vol_Avg_20"]
        volume_breakout = vol_ratio >= 1.5

        if not (strong_candle and volume_breakout):
            return None

        # =========================================================
        # STRATEGY 1: 200-WMA Fresh Reclaim (Cyclical Turnaround)
        # =========================================================
        reclaim_200wma = (
            prev["Close"] < prev["WMA_200"]
            and curr["Close"] > curr["WMA_200"]
            and curr["RSI_14"] >= 55
        )

        # =========================================================
        # STRATEGY 2: Stage-2 Uptrend Breakout (20-Week High)
        # =========================================================
        stage2_breakout = (
            curr["Close"] > curr["WMA_200"]
            and curr["Close"] > curr["WMA_30"]
            and curr["WMA_30"] > df["WMA_30"].iloc[-5]  # Rising 30 WMA
            and prev["Close"] <= prev["High_20W"]
            and curr["Close"] > curr["High_20W"]
            and curr["RSI_14"] >= 60
        )

        # =========================================================
        # STRATEGY 3: 52-Week High Tight Darvas Box Breakout
        # =========================================================
        darvas_52w_breakout = (
            curr["Close"] > curr["WMA_200"]
            and curr["Close"] > curr["WMA_30"]
            and prev["Close"] <= prev["High_52W"]
            and curr["Close"] > curr["High_52W"]
            and curr["Base_Width_4W"] <= 0.18           # Prior 4-week box width <= 18%
            and curr["RSI_14"] >= 60
        )

        if not (reclaim_200wma or stage2_breakout or darvas_52w_breakout):
            return None

        # Build Strategy Tags & Calculate Tailored Stop-Loss
        triggered_setups = []
        if darvas_52w_breakout:
            triggered_setups.append("🔥 52W-High Darvas Box")
        if stage2_breakout:
            triggered_setups.append("📈 Stage-2 (20W High)")
        if reclaim_200wma:
            triggered_setups.append("🔄 200-WMA Reclaim")

        setup_label = " + ".join(triggered_setups)

        # Stop-loss selection (capped at max 12% below entry price)
        if darvas_52w_breakout:
            raw_sl = curr["Low"] * 0.98
        elif stage2_breakout:
            raw_sl = min(curr["Low"], curr["WMA_30"]) * 0.98
        else:
            raw_sl = min(curr["Low"], curr["WMA_200"]) * 0.98

        stop_loss = round(max(float(raw_sl), float(curr["Close"]) * 0.88), 2)
        risk_pct = round(((curr["Close"] - stop_loss) / curr["Close"]) * 100, 1)
        clean_symbol = ticker.replace(".NS", "")

        return {
            "symbol": clean_symbol,
            "setup": setup_label,
            "close": round(float(curr["Close"]), 2),
            "wma_30": round(float(curr["WMA_30"]), 2),
            "wma_200": round(float(curr["WMA_200"]), 2),
            "rsi": round(float(curr["RSI_14"]), 1),
            "vol_multiple": round(float(vol_ratio), 2),
            "base_width_pct": round(float(curr["Base_Width_4W"]) * 100, 1),
            "stop_loss": stop_loss,
            "risk_pct": risk_pct,
            "chart_url": f"https://in.tradingview.com/chart/?symbol=NSE:{clean_symbol}"
        }
    except Exception:
        return None

def send_telegram_chunked(text):
    """Splits long messages safely so Telegram's 4096-character limit is never exceeded."""
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 3800
    chunks = []
    current_chunk = ""

    for block in text.split("\n\n"):
        if len(current_chunk) + len(block) + 2 > max_len:
            chunks.append(current_chunk)
            current_chunk = block + "\n\n"
        else:
            current_chunk += block + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk)

    for chunk in chunks:
        requests.post(tg_url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk.strip(),
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        })

def send_notification(results):
    """Formats and sends the multi-strategy breakout report."""
    date_str = datetime.now().strftime("%d %b %Y")
    if not results:
        message = (
            f"📊 *NSE 3-in-1 Weekly Breakout Scanner ({date_str})*\n\n"
            f"Scan completed successfully ({UNIVERSE_MODE}). No stocks met the strict breakout + 1.5x volume criteria this week."
        )
        print(message)
        send_telegram_chunked(message)
        return

    # Sort results so highest-volume breakouts appear at the very top
    results = sorted(results, key=lambda x: x["vol_multiple"], reverse=True)

    blocks = [f"🚀 *NSE 3-in-1 Weekly Breakout Alerts ({date_str})*\nFound *{len(results)}* stocks sorted by Volume Expansion:"]
    for r in results:
        blocks.append(
            f"📌 *{r['symbol']}* — {r['setup']}\n"
            f"   • Entry (CMP): ₹{r['close']} | Suggested SL: ₹{r['stop_loss']} (-{r['risk_pct']}%)\n"
            f"   • 30 WMA: ₹{r['wma_30']} | 200 WMA: ₹{r['wma_200']}\n"
            f"   • Volume: *{r['vol_multiple']}x Avg* | Weekly RSI: {r['rsi']} | 4W Base: {r['base_width_pct']}%\n"
            f"   • [Open TradingView Chart]({r['chart_url']})"
        )

    full_message = "\n\n".join(blocks)
    print(full_message)
    send_telegram_chunked(full_message)

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"date": date_str, "alerts": results})

if __name__ == "__main__":
    tickers = get_nse_tickers(UNIVERSE_MODE)
    print(f"Scanning {len(tickers)} stocks across all 3 strategies...")
    breakout_stocks = []
    for t in tickers:
        res = scan_stock(t)
        if res:
            breakout_stocks.append(res)

    if breakout_stocks:
        pd.DataFrame(breakout_stocks).to_csv("breakout_alerts.csv", index=False)
    send_notification(breakout_stocks)
