import os
import sys
import io
import math
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# =========================================================
# 1. CREDENTIALS & ₹1,00,000 PORTFOLIO RISK SETTINGS
# =========================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
UNIVERSE_MODE = os.getenv("UNIVERSE_MODE", "NIFTY500")

PORTFOLIO_FILE = "portfolio.txt"
HISTORY_CSV_FILE = "breakout_calls_history.csv"
LATEST_CSV_FILE = "breakout_alerts.csv"

TOTAL_CAPITAL = float(os.getenv("TOTAL_CAPITAL", "100000"))        # ₹1,00,000 Portfolio
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.75"))  # 1.75% Risk (₹1,750)
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "22.0"))      # Max 22% (₹22,000) per stock

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

def calculate_position_size(entry_price, stop_loss):
    """Calculates exact share quantity, Rupee outlay, and pyramid level for ₹1,00,000 capital."""
    max_risk_rupees = TOTAL_CAPITAL * (RISK_PER_TRADE_PCT / 100.0)
    max_alloc_rupees = TOTAL_CAPITAL * (MAX_POSITION_PCT / 100.0)
    risk_per_share = max(entry_price - stop_loss, 0.05)

    raw_shares = math.floor(max_risk_rupees / risk_per_share)
    max_capped_shares = math.floor(max_alloc_rupees / entry_price)
    final_shares = max(1, min(raw_shares, max_capped_shares))

    invested_capital = round(final_shares * entry_price, 2)
    actual_rupee_risk = round(final_shares * risk_per_share, 2)
    actual_risk_pct = round((actual_rupee_risk / TOTAL_CAPITAL) * 100, 2)
    pyramid_trigger = round(entry_price * 1.20, 2)

    return {
        "shares": final_shares,
        "invested_capital": invested_capital,
        "actual_rupee_risk": actual_rupee_risk,
        "actual_risk_pct": actual_risk_pct,
        "pyramid_trigger": pyramid_trigger
    }

def check_portfolio_exits(filepath=PORTFOLIO_FILE):
    """Reads portfolio.txt and checks 30-WMA trailing exits, Stop-Loss, and +20% Pyramid triggers."""
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
            pyramid_price = round(h["buy_price"] * 1.20, 2)

            active_stop = max(h["initial_sl"], wma_30) if cmp_price > h["buy_price"] else h["initial_sl"]

            if weekly_low <= h["initial_sl"] or cmp_price <= h["initial_sl"]:
                status = "🚨 *SELL NOW (Initial Stop-Loss Hit)*"
                action = f"Price breached your initial SL of ₹{h['initial_sl']}. Exit position."
            elif cmp_price < wma_30:
                status = "🚨 *SELL NOW (Closed Below 30-WMA)*"
                action = f"Weekly candle closed at ₹{cmp_price} below 30-WMA (₹{wma_30}). Trend broken."
            elif extension_above_30wma >= 45:
                status = "⚠️ *BOOK 25% PARTIAL PROFIT (Climax Run)*"
                action = f"Stretched +{round(extension_above_30wma, 1)}% above 30-WMA. Trail rest at 10-WMA (₹{wma_10})."
            elif pnl_pct >= 20.0:
                status = "🔥 *MULTIBAGGER ZONE (+20% Reached)*"
                action = f"Move GTT SL to Breakeven (₹{h['buy_price']}) or 30-WMA (₹{wma_30}). Eligible for ₹6,000 Pyramid Add-on."
            else:
                status = "✅ *HOLD (Uptrend Intact)*"
                action = f"Keep GTT SL at ₹{round(active_stop * 0.99, 2)}. Next Pyramid Trigger: ₹{pyramid_price}."

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
    """Downloads weekly data, checks 3 strategies, and builds a full 24-column record."""
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
        candle_date = df.index[-1].strftime("%Y-%m-%d")

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
        clean_setup_csv = setup_label.replace("🔥 ", "").replace("📈 ", "").replace("🔄 ", "")

        if darvas_52w_breakout:
            raw_sl = curr["Low"] * 0.98
        elif stage2_breakout:
            raw_sl = min(curr["Low"], curr["WMA_30"]) * 0.98
        else:
            raw_sl = min(curr["Low"], curr["WMA_200"]) * 0.98

        entry_price = round(float(curr["Close"]), 2)
        stop_loss = round(max(float(raw_sl), entry_price * 0.88), 2)
        risk_pct = round(((entry_price - stop_loss) / entry_price) * 100, 1)
        clean_symbol = ticker.replace(".NS", "")

        sizing = calculate_position_size(entry_price, stop_loss)

        return {
            "Scan_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Candle_Week_Date": candle_date,
            "Symbol": clean_symbol,
            "Universe": UNIVERSE_MODE,
            "Strategy_Triggered": clean_setup_csv,
            "Setup_Display": setup_label,
            "Weekly_Open": round(float(curr["Open"]), 2),
            "Weekly_High": round(float(curr["High"]), 2),
            "Weekly_Low": round(float(curr["Low"]), 2),
            "Entry_Close_CMP": entry_price,
            "WMA_30": round(float(curr["WMA_30"]), 2),
            "WMA_200": round(float(curr["WMA_200"]), 2),
            "Prior_20W_High": round(float(curr["High_20W"]), 2),
            "Prior_52W_High": round(float(curr["High_52W"]), 2),
            "Weekly_Volume": int(curr["Volume"]),
            "Avg_Volume_20W": int(curr["Vol_Avg_20"]),
            "Volume_Multiple": round(float(vol_ratio), 2),
            "Weekly_RSI_14": round(float(curr["RSI_14"]), 1),
            "Base_Width_4W_Pct": round(float(curr["Base_Width_4W"]) * 100, 1),
            "Candle_Close_Pos_Pct": round(float(close_position) * 100, 1),
            "Suggested_SL": stop_loss,
            "SL_Risk_Pct": risk_pct,
            "Rec_Shares_1L": sizing["shares"],
            "Capital_Required_INR": sizing["invested_capital"],
            "Max_Rupee_Risk_INR": sizing["actual_rupee_risk"],
            "Portfolio_Risk_Pct": sizing["actual_risk_pct"],
            "Pyramid_Trigger_20Pct": sizing["pyramid_trigger"],
            "TradingView_URL": f"https://in.tradingview.com/chart/?symbol=NSE:{clean_symbol}"
        }
    except Exception:
        return None

def update_master_csv(results):
    """Saves this week's alerts AND appends them to breakout_calls_history.csv without duplicates."""
    if not results:
        return

    new_df = pd.DataFrame(results).drop(columns=["Setup_Display"], errors="ignore")
    new_df = new_df.sort_values(by="Volume_Multiple", ascending=False)
    new_df.to_csv(LATEST_CSV_FILE, index=False)

    if os.path.exists(HISTORY_CSV_FILE):
        try:
            existing_df = pd.read_csv(HISTORY_CSV_FILE)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Deduplicate by Candle_Week_Date + Symbol so re-runs update rather than duplicate rows
            combined_df = combined_df.drop_duplicates(subset=["Candle_Week_Date", "Symbol"], keep="last")
        except Exception:
            combined_df = new_df
    else:
        combined_df = new_df

    combined_df.to_csv(HISTORY_CSV_FILE, index=False)
    print(f"Master CSV updated: {len(combined_df)} total historical calls stored in {HISTORY_CSV_FILE}.")

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

def send_telegram_csv(filepath, caption):
    """Sends the CSV file directly into the Telegram chat as a downloadable spreadsheet."""
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) or not os.path.exists(filepath):
        return
    tg_doc_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    with open(filepath, "rb") as doc:
        requests.post(
            tg_doc_url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            files={"document": doc}
        )

def send_notification(results, portfolio_reports):
    date_str = datetime.now().strftime("%d %b %Y")
    blocks = []

    if portfolio_reports:
        blocks.append(f"🛡️ *MY PORTFOLIO EXIT & PYRAMID MONITOR ({date_str})*")
        blocks.extend(portfolio_reports)
        blocks.append("──────────────────────────")

    if not results:
        blocks.append(
            f"📊 *NSE 3-in-1 Breakout Scanner ({date_str})*\n"
            f"Capital: ₹{int(TOTAL_CAPITAL):,} | No new stocks met the strict breakout criteria this week."
        )
    else:
        results = sorted(results, key=lambda x: x["Volume_Multiple"], reverse=True)
        blocks.append(
            f"🚀 *NEW BREAKOUT ALERTS ({date_str})*\n"
            f"Capital: *₹{int(TOTAL_CAPITAL):,}* | Target Risk/Trade: *{RISK_PER_TRADE_PCT}% (₹{int(TOTAL_CAPITAL * RISK_PER_TRADE_PCT / 100):,})*"
        )
        for r in results:
            blocks.append(
                f"📌 *{r['Symbol']}* — {r['Setup_Display']}\n"
                f"   • Entry (CMP): ₹{r['Entry_Close_CMP']} | Stop-Loss: ₹{r['Suggested_SL']} (-{r['SL_Risk_Pct']}%)\n"
                f"   • *Buy Qty: {r['Rec_Shares_1L']} shares (₹{r['Capital_Required_INR']:,.0f})*\n"
                f"   • *Max Loss if SL Hits: -₹{r['Max_Rupee_Risk_INR']:,.0f} (-{r['Portfolio_Risk_Pct']}% of ₹1L)*\n"
                f"   • +20% Pyramid Trigger: ₹{r['Pyramid_Trigger_20Pct']} | 30 WMA: ₹{r['WMA_30']}\n"
                f"   • Volume: *{r['Volume_Multiple']}x Avg* | RSI: {r['Weekly_RSI_14']} | 4W Base: {r['Base_Width_4W_Pct']}%\n"
                f"   • [Open TradingView Chart]({r['TradingView_URL']})"
            )

    full_message = "\n\n".join(blocks)
    print(full_message)
    send_telegram_chunked(full_message)

    # Also attach the master cumulative CSV directly in Telegram
    if os.path.exists(HISTORY_CSV_FILE):
        send_telegram_csv(
            HISTORY_CSV_FILE,
            f"📁 Master Breakout Calls Log (Updated {date_str})"
        )

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    portfolio_status = check_portfolio_exits(PORTFOLIO_FILE)

    if mode == "portfolio_only":
        if portfolio_status:
            date_str = datetime.now().strftime("%d %b %Y")
            full_message = f"🛡️ *DAILY PORTFOLIO MONITOR ({date_str})*\n\n" + "\n\n".join(portfolio_status)
            print(full_message)
            send_telegram_chunked(full_message)
    else:
        tickers = get_nse_tickers(UNIVERSE_MODE)
        print(f"Scanning {len(tickers)} stocks with ₹{int(TOTAL_CAPITAL):,} sizing...")
        breakout_stocks = []
        for t in tickers:
            res = scan_stock(t)
            if res:
                breakout_stocks.append(res)

        if breakout_stocks:
            update_master_csv(breakout_stocks)
        send_notification(breakout_stocks, portfolio_status)
