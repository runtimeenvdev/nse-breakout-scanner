import os
import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
UNIVERSE_MODE = os.getenv("UNIVERSE_MODE", "NIFTY500")
PORTFOLIO_FILE = "portfolio.txt"

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
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def check_portfolio_exits(filepath=PORTFOLIO_FILE):
    """Reads portfolio.txt and checks if any owned stock triggered a Sell or Trailing Stop."""
    if not os.path.exists(filepath):
        return []

    holdings = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                holdings.append({
                    "symbol": parts[0].upper().replace(".NS", ""),
                    "buy_price": float(parts[1]),
                    "initial_sl": float(parts[2])
                })

    if not holdings:
        return []

    portfolio_reports = []
    for h in holdings:
        try:
            ticker = f"{h['symbol']}.NS"
            df = yf.download(ticker, period="5y", interval="1wk", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 35:
                continue

            df["WMA_30"] = df["Close"].rolling(window=30).mean()
            df["WMA_10"] = df["Close"].rolling(window=10).mean()
            curr = df.iloc[-1]

            cmp_price = round(float(curr["Close"]), 2)
            weekly_low = round(float(curr["Low"]), 2)
            wma_30 = round(float(curr["WMA_30"]), 2)
            wma_10 = round(float(curr["WMA_10"]), 2)
            pnl_pct = round(((cmp_price - h["buy_price"]) / h["buy_price"]) * 100, 2)
            extension_above_30wma = ((cmp_price - wma_30) / wma_30) * 100

            # Effective trailing stop is the higher of Initial SL or 30 WMA (once in profit)
            active_stop = max(h["initial_sl"], wma_30) if cmp_price > h["buy_price"] else h["initial_sl"]

            # Evaluate Exit Conditions
            if weekly_low <= h["initial_sl"] or cmp_price <= h["initial_sl"]:
                status = "🚨 *SELL NOW (Initial Stop-Loss Hit)*"
                action = f"Price breached your initial SL of ₹{h['initial_sl']}. Exit position."
            elif cmp_price < wma_30:
                status = "🚨 *SELL NOW (Closed Below 30-WMA)*"
                action = f"Weekly candle closed at ₹{cmp_price} below 30-WMA (₹{wma_30}). Trend broken."
            elif extension_above_30wma >= 45:
                status = "⚠️ *BOOK PARTIAL PROFIT (Climax Run)*"
                action = f"Stock is +{round(extension_above_30wma, 1)}% stretched above 30-WMA. Book 25-50% & trail rest at 10-WMA (₹{wma_10})."
            else:
                status = "✅ *HOLD (Uptrend Intact)*"
                action = f"Update your Broker GTT Stop-Loss to ₹{round(active_stop * 0.99, 2)} (just below 30-WMA ₹{wma_30})."

            portfolio_reports.append(
                f"💼 *{h['symbol']}* — {status}\n"
                f"   • Buy: ₹{h['buy_price']} | CMP: ₹{cmp_price} (*{pnl_pct:+}%*)\n"
                f"   • 30-WMA Trail: ₹{wma_30} | Initial SL: ₹{h['initial_sl']}\n"
                f"   • Action: {action}"
            )
        except Exception:
            continue

    return portfolio_reports

def scan_stock(ticker):
    """Downloads weekly data and checks all 3 institutional breakout strategies."""
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
        df["High_52W"] = df["High"].shift(1).rolling(window=52).max()
        df["RSI_14"] = calculate_rsi(df["Close"], period=14)

        prior_4w_high = df["High"].shift(1).rolling(window=4).max()
        prior_4w_low = df["Low"].shift(1).rolling(window=4).min()
        df["Base_Width_4W"] = (prior_4w_high - prior_4w_low) / prior_4w_low

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if curr["Close"] < 50 or curr["Vol_Avg_20"] < 100000:
            return None

        weekly_range = curr["High"] - curr["Low"]
        if weekly_range <= 0:
            return None
        close_position = (curr["Close"] - curr["Low"]) / weekly_range
        strong_candle = close_position >= 0.65

        vol_ratio = curr["Volume"] / curr["Vol_Avg_20"]
        volume_breakout = vol_ratio >= 1.5

        if not (strong_candle and volume_breakout):
            return None

        reclaim_200wma = (
            prev["Close"] < prev["WMA_200"]
            and curr["Close"] > curr["WMA_200"]
            and curr["RSI_14"] >= 55
        )

        stage2_breakout = (
            curr["Close"] > curr["WMA_200"]
            and curr["Close"] > curr["WMA_30"]
            and curr["WMA_30"] > df["WMA_30"].iloc[-5]
            and prev["Close"] <= prev["High_20W"]
            and curr["Close"] > curr["High_20W"]
            and curr["RSI_14"] >= 60
        )

        darvas_52w_breakout = (
            curr["Close"] > curr["WMA_200"]
            and curr["Close"] > curr["WMA_30"]
            and prev["Close"] <= prev["High_52W"]
            and curr["Close"] > curr["High_52W"]
            and curr["Base_Width_4W"] <= 0.18
            and curr["RSI_14"] >= 60
        )

        if not (reclaim_200wma or stage2_breakout or darvas_52w_breakout):
            return None

        triggered_setups = []
        if darvas_52w_breakout:
            triggered_setups.append("🔥 52W-High Darvas Box")
        if stage2_breakout:
            triggered_setups.append("📈 Stage-2 (20W High)")
        if reclaim_200wma:
            triggered_setups.append("🔄 200-WMA Reclaim")

        setup_label = " + ".join(triggered_setups)

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

def send_notification(results, portfolio_reports):
    date_str = datetime.now().strftime("%d %b %Y")
    blocks = []

    # 1. Add Portfolio Sell/Hold Alerts at the very top if portfolio.txt has stocks
    if portfolio_reports:
        blocks.append(f"🛡️ *MY PORTFOLIO EXIT & TRAIL MONITOR ({date_str})*")
        blocks.extend(portfolio_reports)
        blocks.append("──────────────────────────")

    # 2. Add New Buy Breakout Alerts below
    if not results:
        blocks.append(
            f"📊 *NSE 3-in-1 Weekly Breakout Scanner ({date_str})*\n"
            f"No new stocks met the strict breakout + 1.5x volume criteria this week."
        )
    else:
        results = sorted(results, key=lambda x: x["vol_multiple"], reverse=True)
        blocks.append(f"🚀 *NEW WEEKLY BREAKOUT ALERTS ({date_str})*\nFound *{len(results)}* stocks sorted by Volume Expansion:")
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

import sys

if __name__ == "__main__":
    # Check if a mode argument was passed from GitHub Actions
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    print(f"Running in mode: {mode}")
    portfolio_status = check_portfolio_exits(PORTFOLIO_FILE)

    if mode == "portfolio_only":
        # Send ONLY portfolio check to Telegram (runs in ~5 seconds)
        if portfolio_status:
            date_str = datetime.now().strftime("%d %b %Y")
            full_message = f"🛡️ *DAILY PORTFOLIO MONITOR ({date_str})*\n\n" + "\n\n".join(portfolio_status)
            print(full_message)
            send_telegram_chunked(full_message)
        else:
            print("portfolio.txt is empty or missing. No daily check sent.")
    else:
        # Full scan: Portfolio check + 500-stock Friday breakout scan
        tickers = get_nse_tickers(UNIVERSE_MODE)
        print(f"Scanning {len(tickers)} stocks for new breakouts...")
        breakout_stocks = []
        for t in tickers:
            res = scan_stock(t)
            if res:
                breakout_stocks.append(res)

        if breakout_stocks:
            pd.DataFrame(breakout_stocks).to_csv("breakout_alerts.csv", index=False)
        send_notification(breakout_stocks, portfolio_status)
