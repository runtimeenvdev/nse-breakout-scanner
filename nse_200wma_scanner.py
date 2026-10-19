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
UNIVERSE_MODE = os.getenv("UNIVERSE_MODE", "ALL_NSE")

PORTFOLIO_FILE = "portfolio.txt"
HISTORY_CSV_FILE = "breakout_calls_history.csv"
LATEST_CSV_FILE = "breakout_alerts.csv"

TOTAL_CAPITAL = float(os.getenv("TOTAL_CAPITAL", "100000"))        # ₹1,00,000 Portfolio
BASE_RISK_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.75"))     # 1.75% Base Risk (₹1,750 in Bull)
BASE_MAX_POS_PCT = float(os.getenv("MAX_POSITION_PCT", "22.0"))    # 22% Max Slot (₹22,000 in Bull)

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

def get_nifty_regime():
    """Evaluates Nifty 50 (^NSEI) trend regime (10-WMA ~ 50-DMA and 30-WMA ~ 150-DMA)."""
    try:
        nifty = yf.download("^NSEI", period="2y", interval="1wk", progress=False, auto_adjust=True)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)

        nifty["WMA_10"] = nifty["Close"].rolling(window=10).mean()
        nifty["WMA_30"] = nifty["Close"].rolling(window=30).mean()
        curr = nifty.iloc[-1]
        prev_13w = nifty.iloc[-14] if len(nifty) >= 14 else nifty.iloc[0]

        n_close = round(float(curr["Close"]), 2)
        n_wma10 = round(float(curr["WMA_10"]), 2)
        n_wma30 = round(float(curr["WMA_30"]), 2)
        nifty_13w_ret = float(curr["Close"] / prev_13w["Close"])

        if n_close >= n_wma10 and n_close >= n_wma30:
            return {
                "regime": "BULL",
                "banner": f"🟢 *MARKET REGIME: BULLISH* (Nifty ₹{n_close} > 10-WMA ₹{n_wma10} & 30-WMA ₹{n_wma30})",
                "risk_mult": 1.0,      # Full 1.75% risk (₹1,750)
                "min_rs": 1.00,        # Stock must beat or match Nifty 50
                "nifty_13w_ret": nifty_13w_ret
            }
        elif n_close < n_wma10 and n_close >= n_wma30:
            return {
                "regime": "CAUTION",
                "banner": f"🟡 *MARKET REGIME: CHOPPY / PULLBACK* (Nifty ₹{n_close} below 10-WMA ₹{n_wma10}; Risk Halved to 0.88%)",
                "risk_mult": 0.5,      # Half risk: 0.875% (₹875), Max slot ₹11,000
                "min_rs": 1.05,        # Only buy leaders beating Nifty by +5%+
                "nifty_13w_ret": nifty_13w_ret
            }
        else:
            return {
                "regime": "BEAR",
                "banner": f"🔴 *MARKET REGIME: BEARISH* (Nifty ₹{n_close} below 30-WMA ₹{n_wma30}; Cash Protection Mode - Watchlist Only)",
                "risk_mult": 0.0,      # 0 shares recommended; preserve ₹1,00,000 in cash/LIQUIDBEES
                "min_rs": 1.08,
                "nifty_13w_ret": nifty_13w_ret
            }
    except Exception:
        return {
            "regime": "BULL",
            "banner": "🟢 *MARKET REGIME: DEFAULT* (Nifty check unavailable)",
            "risk_mult": 1.0,
            "min_rs": 1.00,
            "nifty_13w_ret": 1.0
        }

def calculate_position_size(entry_price, stop_loss, regime_mult, is_only_reclaim, is_extended):
    """Computes shares & Rupee allocation adjusted for Market Regime, Reclaim cap, and Extension."""
    if regime_mult <= 0.0:
        return {
            "shares": 0,
            "invested_capital": 0.0,
            "actual_rupee_risk": 0.0,
            "actual_risk_pct": 0.0,
            "pyramid_trigger": round(entry_price * 1.20, 2)
        }

    # Cap pure 200-WMA Reclaims at 12% (₹12,000) and 1.25% risk to protect against overhead supply
    effective_risk_pct = (1.25 if is_only_reclaim else BASE_RISK_PCT) * regime_mult
    effective_max_pos_pct = (12.0 if is_only_reclaim else BASE_MAX_POS_PCT) * regime_mult

    # If candle is extended (5% to 12% above pivot), recommend a 50% starter on Friday
    if is_extended:
        effective_risk_pct *= 0.5
        effective_max_pos_pct *= 0.5

    max_risk_rupees = TOTAL_CAPITAL * (effective_risk_pct / 100.0)
    max_alloc_rupees = TOTAL_CAPITAL * (effective_max_pos_pct / 100.0)
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

def scan_stock(ticker, market_info):
    """Downloads weekly data and applies Regime, Relative Strength, Extension, and Overhead filters."""
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
        prev_13w = df.iloc[-14]
        candle_date = df.index[-1].strftime("%Y-%m-%d")

        if curr["Close"] < 50 or curr["Vol_Avg_20"] < 100000:
            return None

        # Fix 2: Relative Strength (13-Week performance vs. Nifty 50)
        stock_13w_ret = float(curr["Close"] / prev_13w["Close"])
        rs_vs_nifty = round(stock_13w_ret / market_info["nifty_13w_ret"], 2)
        if rs_vs_nifty < market_info["min_rs"]:
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

        # =========================================================
        # FIX 3: UPGRADED 200-WMA RECLAIM (WITH OVERHEAD CLEARANCE)
        # Requires rising 30-WMA, Close > 20W High, and within 25% of 52W High
        # =========================================================
        reclaim_200wma = (
            market_info["regime"] == "BULL"              # Disable bottom-fishing in choppy/bear regimes
            and prev["Close"] < prev["WMA_200"]
            and curr["Close"] > curr["WMA_200"]
            and curr["Close"] > curr["WMA_30"]
            and curr["WMA_30"] > df["WMA_30"].iloc[-5]   # Rising 30-WMA
            and curr["Close"] > curr["High_20W"]         # Cleared 5 months of overhead supply
            and curr["Close"] >= (0.75 * curr["High_52W"])
            and curr["RSI_14"] >= 58
        )

        # =========================================================
        # STRATEGY 2: Stage-2 Uptrend Breakout (20-Week High)
        # =========================================================
        stage2_breakout = (
            curr["Close"] > curr["WMA_200"]
            and curr["Close"] > curr["WMA_30"]
            and curr["WMA_30"] > df["WMA_30"].iloc[-5]
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
            and curr["Base_Width_4W"] <= 0.18
            and curr["RSI_14"] >= 60
        )

        if not (reclaim_200wma or stage2_breakout or darvas_52w_breakout):
            return None

        # =========================================================
        # FIX 1: PIVOT EXTENSION & LATE-CHASING FILTER
        # =========================================================
        if darvas_52w_breakout:
            pivot_price = float(curr["High_52W"])
            raw_sl = curr["Low"] * 0.98
        elif stage2_breakout:
            pivot_price = float(curr["High_20W"])
            raw_sl = min(curr["Low"], curr["WMA_30"]) * 0.98
        else:
            pivot_price = float(max(curr["WMA_200"], curr["High_20W"]))
            raw_sl = min(curr["Low"], curr["WMA_200"]) * 0.98

        entry_price = round(float(curr["Close"]), 2)
        weekly_gain_pct = round(((entry_price - float(prev["Close"])) / float(prev["Close"])) * 100, 1)
        pivot_ext_pct = round(((entry_price - pivot_price) / pivot_price) * 100, 1)

        # Reject severely overextended parabolic spikes (>12% above pivot or >16% weekly jump)
        if pivot_ext_pct > 12.0 or weekly_gain_pct > 16.0:
            return None

        is_extended = (pivot_ext_pct > 5.0) or (weekly_gain_pct > 8.5)
        pullback_buy_price = round(pivot_price + 0.40 * (entry_price - pivot_price), 2)

        if market_info["regime"] == "BEAR":
            entry_plan = f"🔴 WATCHLIST ONLY (Bear Regime — Wait for Nifty > 10-WMA)"
        elif is_extended:
            entry_plan = f"⏳ EXTENDED (+{pivot_ext_pct}% > Pivot) — Buy 50% Now OR Place GTT Buy @ ₹{pullback_buy_price}"
        else:
            entry_plan = f"✅ ACTIONABLE NOW (+{pivot_ext_pct}% Tight Breakout above ₹{round(pivot_price, 2)})"

        triggered_setups = []
        if darvas_52w_breakout:
            triggered_setups.append("🔥 52W-High Darvas Box")
        if stage2_breakout:
            triggered_setups.append("📈 Stage-2 (20W High)")
        if reclaim_200wma:
            triggered_setups.append("🔄 200-WMA + 20W-High Turnaround")

        setup_label = " + ".join(triggered_setups)
        clean_setup_csv = setup_label.replace("🔥 ", "").replace("📈 ", "").replace("🔄 ", "")
        is_only_reclaim = reclaim_200wma and not (darvas_52w_breakout or stage2_breakout)

        stop_loss = round(max(float(raw_sl), entry_price * 0.88), 2)
        risk_pct = round(((entry_price - stop_loss) / entry_price) * 100, 1)
        clean_symbol = ticker.replace(".NS", "")

        sizing = calculate_position_size(
            entry_price, stop_loss, market_info["risk_mult"], is_only_reclaim, is_extended
        )

        return {
            "Scan_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Candle_Week_Date": candle_date,
            "Market_Regime": market_info["regime"],
            "Symbol": clean_symbol,
            "Universe": UNIVERSE_MODE,
            "Strategy_Triggered": clean_setup_csv,
            "Setup_Display": setup_label,
            "Entry_Plan": entry_plan,
            "Weekly_Open": round(float(curr["Open"]), 2),
            "Weekly_High": round(float(curr["High"]), 2),
            "Weekly_Low": round(float(curr["Low"]), 2),
            "Entry_Close_CMP": entry_price,
            "Breakout_Pivot": round(pivot_price, 2),
            "Pivot_Extension_Pct": pivot_ext_pct,
            "Weekly_Gain_Pct": weekly_gain_pct,
            "Pullback_GTT_Buy": pullback_buy_price,
            "RS_vs_Nifty_13W": rs_vs_nifty,
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

    new_df = pd.DataFrame(results).drop(columns=["Setup_Display", "Entry_Plan"], errors="ignore")
    new_df = new_df.sort_values(by=["Pivot_Extension_Pct", "Volume_Multiple"], ascending=[True, False])
    new_df.to_csv(LATEST_CSV_FILE, index=False)

    if os.path.exists(HISTORY_CSV_FILE):
        try:
            existing_df = pd.read_csv(HISTORY_CSV_FILE)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
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
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) or not os.path.exists(filepath):
        return
    tg_doc_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    with open(filepath, "rb") as doc:
        requests.post(
            tg_doc_url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            files={"document": doc}
        )

def send_notification(results, portfolio_reports, market_info):
    date_str = datetime.now().strftime("%d %b %Y")
    blocks = [market_info["banner"]]

    if portfolio_reports:
        blocks.append(f"🛡️ *MY PORTFOLIO EXIT & PYRAMID MONITOR ({date_str})*")
        blocks.extend(portfolio_reports)
        blocks.append("──────────────────────────")

    if not results:
        blocks.append(
            f"📊 *NSE Regime-Filtered Breakout Scanner ({date_str})*\n"
            f"Capital: ₹{int(TOTAL_CAPITAL):,} | No stocks passed the Pivot Extension, RS >= {market_info['min_rs']}, and Volume filters this week."
        )
    else:
        # Prioritize tight, actionable breakouts first, then sort by highest Volume Multiple
        results = sorted(results, key=lambda x: (x["Pivot_Extension_Pct"] > 5.0, -x["Volume_Multiple"]))
        blocks.append(
            f"🚀 *NEW BREAKOUT ALERTS ({date_str})*\n"
            f"Capital: *₹{int(TOTAL_CAPITAL):,}* | Active Regime Risk Multiplier: *{market_info['risk_mult']}x*"
        )
        for r in results:
            blocks.append(
                f"📌 *{r['Symbol']}* — {r['Setup_Display']}\n"
                f"   • {r['Entry_Plan']}\n"
                f"   • CMP: ₹{r['Entry_Close_CMP']} (+{r['Weekly_Gain_Pct']}% wk) | Pivot: ₹{r['Breakout_Pivot']} | SL: ₹{r['Suggested_SL']} (-{r['SL_Risk_Pct']}%)\n"
                f"   • *Friday Buy Qty: {r['Rec_Shares_1L']} shares (₹{r['Capital_Required_INR']:,.0f})* | Risk: -₹{r['Max_Rupee_Risk_INR']:,.0f} (-{r['Portfolio_Risk_Pct']}%)\n"
                f"   • RS vs Nifty (13W): *{r['RS_vs_Nifty_13W']}x* | Vol: *{r['Volume_Multiple']}x* | 4W Base: {r['Base_Width_4W_Pct']}%\n"
                f"   • [Open TradingView Chart]({r['TradingView_URL']})"
            )

    full_message = "\n\n".join(blocks)
    print(full_message)
    send_telegram_chunked(full_message)

    if os.path.exists(HISTORY_CSV_FILE):
        send_telegram_csv(
            HISTORY_CSV_FILE,
            f"📁 Master Breakout Calls Log (Updated {date_str} | Regime: {market_info['regime']})"
        )

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    market_info = get_nifty_regime()
    portfolio_status = check_portfolio_exits(PORTFOLIO_FILE)

    if mode == "portfolio_only":
        if portfolio_status:
            date_str = datetime.now().strftime("%d %b %Y")
            full_message = f"{market_info['banner']}\n\n🛡️ *DAILY PORTFOLIO MONITOR ({date_str})*\n\n" + "\n\n".join(portfolio_status)
            print(full_message)
            send_telegram_chunked(full_message)
    else:
        tickers = get_nse_tickers(UNIVERSE_MODE)
        print(f"Regime: {market_info['regime']} | Scanning {len(tickers)} stocks...")
        breakout_stocks = []
        for t in tickers:
            res = scan_stock(t, market_info)
            if res:
                breakout_stocks.append(res)

        if breakout_stocks:
            update_master_csv(breakout_stocks)
        send_notification(breakout_stocks, portfolio_status, market_info)
