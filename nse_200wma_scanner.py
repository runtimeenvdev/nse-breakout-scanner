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
    """Downloads weekly data and checks 200 WMA Reclaim and Stage 2 Uptrend Breakouts."""
    try:
        df = yf.download(ticker, period="5y", interval="1wk", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 205:
            return None

        df["WMA_200"] = df["Close"].rolling(window=200).mean()
        df["WMA_30"] = df["Close"].rolling(window=30).mean()
        df["Vol_Avg_20"] = df["Volume"].rolling(window=20).mean()
        df["High_20W"] = df["High"].shift(1).rolling(window=20).max()
        df["RSI_14"] = calculate_rsi(df["Close"], period=14)

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Liquidity filter: Price >= Rs 50 and 20-week avg volume >= 1,00,000
        if curr["Close"] < 50 or curr["Vol_Avg_20"] < 100000:
            return None

        weekly_range = curr["High"] - curr["Low"]
        if weekly_range <= 0:
            return None
        close_position = (curr["Close"] - curr["Low"]) / weekly_range
        strong_candle = close_position >= 0.65

        vol_ratio = curr["Volume"] / curr["Vol_Avg_20"]
        volume_breakout = vol_ratio >= 1.5
        momentum_ok = curr["RSI_14"] >= 55

        # Setup 1: Fresh 200 WMA Crossover / Reclaim
        reclaim_200wma = (
            prev["Close"] < prev["WMA_200"]
            and curr["Close"] > curr["WMA_200"]
            and volume_breakout
            and strong_candle
            and momentum_ok
        )

        # Setup 2: Uptrend Breakout (Above 200 WMA & 30 WMA + New 20-Week High)
        uptrend_breakout = (
            curr["Close"] > curr["WMA_200"]
            and curr["Close"] > curr["WMA_30"]
            and curr["WMA_30"] > df["WMA_30"].iloc[-5]
            and prev["Close"] <= prev["High_20W"]
            and curr["Close"] > curr["High_20W"]
            and volume_breakout
            and strong_candle
            and curr["RSI_14"] >= 60
        )

        if reclaim_200wma or uptrend_breakout:
            setup_type = "200 WMA Fresh Reclaim" if reclaim_200wma else "Stage 2 Uptrend Breakout"
            stop_loss = round(min(curr["Low"], curr["WMA_30"] if uptrend_breakout else curr["WMA_200"]) * 0.98, 2)
            clean_symbol = ticker.replace(".NS", "")
            return {
                "symbol": clean_symbol,
                "setup": setup_type,
                "close": round(curr["Close"], 2),
                "wma_200": round(curr["WMA_200"], 2),
                "rsi": round(curr["RSI_14"], 1),
                "vol_multiple": round(vol_ratio, 2),
                "stop_loss": stop_loss,
                "chart_url": f"https://in.tradingview.com/chart/?symbol=NSE:{clean_symbol}"
            }
        return None
    except Exception:
        return None

def send_notification(results):
    """Sends a Telegram notification even if 0 stocks matched so you know the bot ran."""
    date_str = datetime.now().strftime("%d %b %Y")
    if not results:
        message = f"📊 *NSE 200-Week Scanner ({date_str})*\n\nScan completed successfully. No stocks met all strict breakout + 1.5x volume criteria this week."
    else:
        lines = [f"🚀 *NSE 200-Week Breakout Alerts ({date_str})*\n"]
        for r in results:
            lines.append(
                f"📌 *{r['symbol']}* ({r['setup']})\n"
                f"   • Entry (CMP): ₹{r['close']} | 200 WMA: ₹{r['wma_200']}\n"
                f"   • Volume: {r['vol_multiple']}x Avg | Weekly RSI: {r['rsi']}\n"
                f"   • Suggested SL: ₹{r['stop_loss']}\n"
                f"   • [Open TradingView Chart]({r['chart_url']})\n"
            )
        message = "\n".join(lines)

    print(message)

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(tg_url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        })

if __name__ == "__main__":
    tickers = get_nse_tickers(UNIVERSE_MODE)
    print(f"Scanning {len(tickers)} stocks...")
    breakout_stocks = []
    for t in tickers:
        res = scan_stock(t)
        if res:
            breakout_stocks.append(res)
    if breakout_stocks:
        pd.DataFrame(breakout_stocks).to_csv("breakout_alerts.csv", index=False)
    send_notification(breakout_stocks)
