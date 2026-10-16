import os
import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "100"))  # Number of NSE stocks to backtest

def get_nifty_tickers(limit=100):
    """Fetches liquid NSE stocks from Nifty 500 for historical backtesting."""
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    res = requests.get(url, headers=headers, timeout=15)
    df = pd.read_csv(io.StringIO(res.text))
    symbols = df["Symbol"].dropna().unique().tolist()[:limit]
    return [f"{sym}.NS" for sym in symbols]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def prepare_indicators(df):
    """Computes weekly moving averages, volume ratios, 20W/52W highs, and consolidation tightness."""
    df["WMA_200"] = df["Close"].rolling(window=200).mean()
    df["WMA_30"] = df["Close"].rolling(window=30).mean()
    df["Vol_Avg_20"] = df["Volume"].rolling(window=20).mean()
    df["High_20W"] = df["High"].shift(1).rolling(window=20).max()
    df["High_52W"] = df["High"].shift(1).rolling(window=52).max()
    df["RSI_14"] = calculate_rsi(df["Close"], period=14)

    # Prior 4-week consolidation range width (for Darvas / 52W High Tight Base)
    prior_4w_high = df["High"].shift(1).rolling(window=4).max()
    prior_4w_low = df["Low"].shift(1).rolling(window=4).min()
    df["Base_Width_4W"] = (prior_4w_high - prior_4w_low) / prior_4w_low
    return df

def simulate_strategy_on_stock(df, ticker, strategy_name):
    """Runs week-by-week historical simulation on a single stock for a chosen strategy."""
    trades = []
    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    entry_date = None
    entry_idx = 0

    for i in range(205, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        date = df.index[i]

        # -----------------------------------------------------
        # MANAGE OPEN TRADE (EXIT LOGIC)
        # -----------------------------------------------------
        if in_trade:
            weeks_held = i - entry_idx
            exit_price = None
            exit_reason = ""

            # 1. Hard Stop-Loss hit intra-week
            if curr["Low"] <= stop_loss:
                exit_price = stop_loss
                exit_reason = "Stop-Loss"
            # 2. Trailing Stop: Weekly close below 30 WMA (after holding at least 4 weeks)
            elif weeks_held >= 4 and curr["Close"] < curr["WMA_30"]:
                exit_price = curr["Close"]
                exit_reason = "30-WMA Trail"
            # 3. Time Stop: 52 weeks max hold or end of dataset
            elif weeks_held >= 52 or i == len(df) - 1:
                exit_price = curr["Close"]
                exit_reason = "52W Time Exit"

            if exit_price is not None:
                pct_return = ((exit_price - entry_price) / entry_price) * 100.0
                risk_pct = ((entry_price - stop_loss) / entry_price) * 100.0
                r_multiple = pct_return / risk_pct if risk_pct > 0 else 0.0

                trades.append({
                    "Symbol": ticker.replace(".NS", ""),
                    "Strategy": strategy_name,
                    "Entry_Date": entry_date.strftime("%Y-%m-%d"),
                    "Exit_Date": date.strftime("%Y-%m-%d"),
                    "Weeks_Held": weeks_held,
                    "Entry_Price": round(entry_price, 2),
                    "Exit_Price": round(exit_price, 2),
                    "Return_Pct": round(pct_return, 2),
                    "R_Multiple": round(r_multiple, 2),
                    "Exit_Reason": exit_reason
                })
                in_trade = False
            continue

        # -----------------------------------------------------
        # CHECK ENTRY CONDITIONS (WHEN NOT IN A TRADE)
        # -----------------------------------------------------
        if curr["Close"] < 50 or curr["Vol_Avg_20"] < 100000:
            continue

        weekly_range = curr["High"] - curr["Low"]
        if weekly_range <= 0:
            continue

        close_pos = (curr["Close"] - curr["Low"]) / weekly_range
        strong_candle = close_pos >= 0.65
        vol_breakout = (curr["Volume"] / curr["Vol_Avg_20"]) >= 1.5

        if not (strong_candle and vol_breakout):
            continue

        signal = False
        raw_stop = curr["Low"] * 0.98

        if strategy_name == "1. 200-WMA Reclaim":
            signal = (
                prev["Close"] < prev["WMA_200"]
                and curr["Close"] > curr["WMA_200"]
                and curr["RSI_14"] >= 55
            )
            raw_stop = min(curr["Low"], curr["WMA_200"]) * 0.98

        elif strategy_name == "2. Stage-2 (20W High)":
            signal = (
                curr["Close"] > curr["WMA_200"]
                and curr["Close"] > curr["WMA_30"]
                and curr["WMA_30"] > df["WMA_30"].iloc[i - 4]
                and prev["Close"] <= prev["High_20W"]
                and curr["Close"] > curr["High_20W"]
                and curr["RSI_14"] >= 60
            )
            raw_stop = min(curr["Low"], curr["WMA_30"]) * 0.98

        elif strategy_name == "3. 52W-High Darvas Box":
            signal = (
                curr["Close"] > curr["WMA_200"]
                and curr["Close"] > curr["WMA_30"]
                and prev["Close"] <= prev["High_52W"]
                and curr["Close"] > curr["High_52W"]
                and curr["Base_Width_4W"] <= 0.18  # Tight <18% 4-week box before breakout
                and curr["RSI_14"] >= 60
            )
            raw_stop = curr["Low"] * 0.98

        if signal:
            in_trade = True
            entry_price = float(curr["Close"])
            # Cap maximum initial stop-loss distance at 12% below entry
            stop_loss = max(float(raw_stop), entry_price * 0.88)
            entry_date = date
            entry_idx = i

    return trades

def summarize_performance(trades_df):
    """Calculates institutional backtest metrics for each strategy."""
    summary_rows = []
    for strat, group in trades_df.groupby("Strategy"):
        total = len(group)
        wins = group[group["Return_Pct"] > 0]
        losses = group[group["Return_Pct"] <= 0]

        win_rate = (len(wins) / total) * 100 if total > 0 else 0
        avg_return = group["Return_Pct"].mean() if total > 0 else 0
        median_return = group["Return_Pct"].median() if total > 0 else 0
        avg_win = wins["Return_Pct"].mean() if len(wins) > 0 else 0
        avg_loss = losses["Return_Pct"].mean() if len(losses) > 0 else 0
        max_win = group["Return_Pct"].max() if total > 0 else 0
        avg_hold = group["Weeks_Held"].mean() if total > 0 else 0

        gross_profit = wins["Return_Pct"].sum()
        gross_loss = abs(losses["Return_Pct"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

        summary_rows.append({
            "Strategy": strat,
            "Trades": total,
            "Win_Rate_Pct": round(win_rate, 1),
            "Avg_Return_Pct": round(avg_return, 2),
            "Median_Ret_Pct": round(median_return, 2),
            "Avg_Win_Pct": round(avg_win, 2),
            "Avg_Loss_Pct": round(avg_loss, 2),
            "Profit_Factor": round(profit_factor, 2),
            "Best_Trade_Pct": round(max_win, 1),
            "Avg_Weeks_Held": round(avg_hold, 1)
        })
    return pd.DataFrame(summary_rows)

def send_telegram_report(summary_df, stock_count):
    lines = [f"📊 *10-Year NSE Strategy Backtest ({stock_count} Stocks)*\n"]
    for _, row in summary_df.iterrows():
        lines.append(
            f"🏆 *{row['Strategy']}*\n"
            f"   • Total Trades: {row['Trades']} | Win Rate: *{row['Win_Rate_Pct']}%*\n"
            f"   • Avg Return/Trade: *{row['Avg_Return_Pct']}%* (Median: {row['Median_Ret_Pct']}%)\n"
            f"   • Avg Win: +{row['Avg_Win_Pct']}% | Avg Loss: {row['Avg_Loss_Pct']}%\n"
            f"   • Profit Factor: *{row['Profit_Factor']}* | Best Trade: +{row['Best_Trade_Pct']}%\n"
            f"   • Avg Hold Time: {row['Avg_Weeks_Held']} weeks\n"
        )
    message = "\n".join(lines)
    print(message)

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        })

if __name__ == "__main__":
    tickers = get_nifty_tickers(SAMPLE_SIZE)
    print(f"Downloading 10-year weekly history for {len(tickers)} NSE stocks...")

    strategies = [
        "1. 200-WMA Reclaim",
        "2. Stage-2 (20W High)",
        "3. 52W-High Darvas Box"
    ]
    all_trades = []

    for idx, t in enumerate(tickers, 1):
        try:
            df = yf.download(t, period="10y", interval="1wk", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 220:
                continue
            df = prepare_indicators(df)
            for strat in strategies:
                trades = simulate_strategy_on_stock(df, t, strat)
                all_trades.extend(trades)
        except Exception:
            continue

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("backtest_all_trades.csv", index=False)

        summary_df = summarize_performance(trades_df)
        summary_df.to_csv("backtest_summary.csv", index=False)
        send_telegram_report(summary_df, len(tickers))
    else:
        print("No historical trades found.")
