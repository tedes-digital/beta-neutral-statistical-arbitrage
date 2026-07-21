import yfinance as yf
import pandas as pd
import itertools
import time
import os
import logging
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import requests

from io import StringIO


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

 # Download and parse the current S&P 500 constituent tickers from Wikipedia
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    sp500_table = None
    for table in tables:
        if 'Symbol' in table.columns:
            sp500_table = table
            break

    if sp500_table is None:
        raise ValueError('Could not find the S&P 500 constituents table with a Symbol column.')

    tickers = sp500_table['Symbol'].astype(str).tolist()
    tickers = [t.replace('.', '-') for t in tickers]

    logging.info(f'Retrieved {len(tickers)} S&P 500 tickers from Wikipedia.')
    return tickers

 # Download historical prices for each ticker with retry handling
def download_stock_data(tickers, start_date, end_date, interval="1d", retries=3):
    all_data = {}
    for ticker in tickers:
        for attempt in range(1, retries+1):
            try:
                logging.info(f"Downloading {ticker} (attempt {attempt})")
                df = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )
                if 'Close' in df and not df['Close'].empty:
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)
                    all_data[ticker] = df[['Close']]
                else:
                    logging.warning(f"{ticker}: no Close data, skipping.")
                break  # exit retry loop on success or valid empty
            except Exception as e:
                logging.error(f"{ticker} download failed: {e}")
                time.sleep(2 ** attempt)  # exponential backoff
        else:
            logging.error(f"{ticker} skipped after {retries} attempts")
    logging.info(f"Downloaded data for {len(all_data)}/{len(tickers)} tickers")
    return all_data
                

 # Save each ticker's closing-price series as a separate CSV file
def save_to_csv(data, folder="price_data"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for ticker, df in data.items():
        # 1) Extract only the Close column
        df2 = df[['Close']].copy()
        # 2) Rename it to “Close price”
        df2.columns = ['Close price']
        # 3) Name the index “Date”
        df2.index.name = 'Date'

        # 4) Write out with semicolon separator
        file_path = os.path.join(folder, f"{ticker}.csv")
        df2.to_csv(file_path, sep=';', index=True)

        logging.info(f"Saved {ticker} data to {file_path}")

 # Load saved price files and combine them into a single ordered price matrix
def prepare_for_cointegration(data_folder="price_data"):
    close_series = {}
    for fname in os.listdir(data_folder):
        if not fname.endswith(".csv"):
            continue
        ticker = fname.replace(".csv","")
        path   = os.path.join(data_folder, fname)
        df = pd.read_csv(path, sep=';', index_col=0, parse_dates=True)
        if "Close price" not in df.columns or df["Close price"].empty:
            continue
        s = df["Close price"].copy()
        s.sort_index(inplace=True)
        close_series[ticker] = s

    # outer join preserves the available history of each ticker.
    # Complete columns are selected later after the IS/OOS split is defined.
    merged = pd.concat(close_series.values(), axis=1, join="outer")
    merged.columns = close_series.keys()
    merged.sort_index(inplace=True)
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged


 # Keep only tickers with complete observations across the full IS/OOS horizon
def filter_complete_price_history(prices, start_date, end_date):
    period_prices = prices.loc[(prices.index >= pd.to_datetime(start_date)) & (prices.index < pd.to_datetime(end_date))].copy()

    if period_prices.empty:
        raise ValueError(
            f"No price data available between {start_date} and {end_date}. "
            f"Available data range: {prices.index.min()} to {prices.index.max()}"
        )

    complete_columns = period_prices.columns[period_prices.notna().all()].tolist()
    removed = len(period_prices.columns) - len(complete_columns)
    logging.info(f"Keeping {len(complete_columns)} tickers with complete history; removed {removed} tickers with missing data.")

    if not complete_columns:
        raise ValueError("No tickers have complete price history for the selected IS/OOS period.")

    return period_prices[complete_columns]


 # Split the filtered price matrix into in-sample and out-of-sample datasets
def split_prices_by_period(prices, is_start, is_end, oos_start, oos_end):
    prices_is = prices.loc[(prices.index >= pd.to_datetime(is_start)) & (prices.index < pd.to_datetime(is_end))].copy()
    prices_oos = prices.loc[(prices.index >= pd.to_datetime(oos_start)) & (prices.index < pd.to_datetime(oos_end))].copy()

    if prices_is.empty:
        raise ValueError("In-sample dataset is empty. Check IS dates.")
    if prices_oos.empty:
        raise ValueError("Out-of-sample dataset is empty. Check OOS dates.")

    return prices_is, prices_oos


 # Calculate the annualised Sharpe ratio from a portfolio value series
def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0, periods_per_year=252):
    values = pd.Series(portfolio_values).dropna()
    returns = values.pct_change().dropna()

    if returns.empty or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    return (excess_returns.mean() / returns.std()) * (periods_per_year ** 0.5)


 # Regress one series on another and test the residuals for stationarity
def engle_granger_test(series_x, series_y, significance=0.05):
    # Step 1: regress Y on X
    model = sm.OLS(series_y, sm.add_constant(series_x)).fit()
    resid = model.resid
    # Step 2: ADF on residuals
    pvalue = adfuller(resid, autolag='AIC')[1]
    return pvalue

if __name__ == "__main__":
    # Ensure all relative paths are resolved from the project folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # === Ticker retrieval: Download S&P 500 tickers ===
    import shutil
    if os.path.exists("price_data"):
        shutil.rmtree("price_data")
    if os.path.exists("signals"):
        shutil.rmtree("signals")
    if os.path.exists("trading_results"):
        shutil.rmtree("trading_results")
    tickers = get_sp500_tickers()[:503]  # Full S&P 500 universe; incomplete histories are filtered later

    # === In-sample / out-of-sample setup ===
    # In-sample: model construction (ADF, Engle-Granger, beta, weights)
    # Out-of-sample: trading simulation and performance evaluation
    is_start = "2023-01-01"
    is_end = "2024-01-01"
    oos_start = "2024-01-01"
    oos_end = "2025-01-01"

    # Full download period must cover both IS and OOS
    start_date = is_start
    end_date = oos_end

    # === Data download: Retrieve historical prices for all tickers ===
    stock_data = download_stock_data(tickers, start_date, end_date)

    # === CSV export: Save each ticker's price history ===
    save_to_csv(stock_data)

    # === Price-matrix preparation: Build wide price matrix from CSVs ===
    prices = prepare_for_cointegration("price_data")
    print(prices.shape)        # raw matrix before filtering
    print(prices.head())
    print(f"Raw price data range: {prices.index.min()} to {prices.index.max()}")

    # === Complete-history filtering: Keep only tickers with full IS/OOS data ===
    prices = filter_complete_price_history(prices, is_start, oos_end)
    print(f"Filtered price matrix shape: {prices.shape}")
    print(f"Filtered price data range: {prices.index.min()} to {prices.index.max()}")

    # === IS/OOS split: Separate in-sample and out-of-sample datasets ===
    prices_is, prices_oos = split_prices_by_period(prices, is_start, is_end, oos_start, oos_end)
    print(f"In-sample price matrix shape: {prices_is.shape}")
    print(f"Out-of-sample price matrix shape: {prices_oos.shape}")

    # === ADF testing: Find non-stationary tickers for cointegration ===
    non_stationary = []   # Collect tickers eligible for cointegration
    for ticker in prices_is.columns:
        series = prices_is[ticker]
        result = adfuller(series, autolag='AIC')
        pvalue = result[1]
        print(f"{ticker:6s} ADF p‑value = {pvalue:.4f}")
        if pvalue > 0.05:
            non_stationary.append(ticker)
    total = len(prices_is.columns)
    passed = len(non_stationary)
    print(f"\n{passed} out of {total} tickers are non-stationary (p > 0.05)")
    print("\nTickers with p‑value > 0.05 (non‑stationary):")
    print(non_stationary)

    # === Engle-Granger testing: Identify cointegrated pairs ===
    eg_results = []
    for x, y in itertools.combinations(non_stationary, 2):
        pval = engle_granger_test(prices_is[x], prices_is[y])
        eg_results.append({'pair': (x, y), 'pvalue': pval})
    eg_df = pd.DataFrame(eg_results)
    eg_df['cointegrated'] = eg_df['pvalue'] <= 0.05
    # Filter & sort
    cointegrated_df = eg_df[eg_df['cointegrated']].sort_values('pvalue')
    # Print and save
    print("\nAll cointegrated pairs (p ≤ 0.05):")
    print(cointegrated_df.to_string(index=False))
    cointegrated_df.to_csv("cointegrated_pairs.csv", index=False)
    print("Saved all cointegrated pairs to cointegrated_pairs.csv")

    # === Beta estimation: Estimate stock betas to market on IS data ===
    market = yf.download('^GSPC', start=is_start, end=is_end, progress=False)
    market_returns = market[['Close']].pct_change().dropna()
    market_returns.columns = ['MKT']
    rets = prices_is.pct_change().dropna()
    data = pd.concat([rets, market_returns], axis=1).dropna()
    betas = {}
    for t in data.columns.drop("MKT"):
        mdl = sm.OLS(data[t], sm.add_constant(data["MKT"])).fit()
        betas[t] = mdl.params["MKT"]

    # === Beta-neutral weight construction: For each cointegrated pair ===
    weights = []
    for row in eg_df[eg_df.cointegrated].itertuples():
        x, y = row.pair
        bx, by = betas[x], betas[y]
        # solve: bx*w_x + by*w_y = 0 and w_x + w_y = 1
        w_x =  by / (by - bx)
        w_y = -bx / (by - bx)
        weights.append({"pair": (x, y), "w_x": w_x, "w_y": w_y})
    w_df = pd.DataFrame(weights)
    print("\nBeta-neutral weights:")
    print(w_df)
    # Export weights
    output_file = os.path.join("beta_weights", "beta_neutral_weights.csv")
    if not os.path.exists("beta_weights"):
        os.makedirs("beta_weights")
    w_df.to_csv(output_file, index=False, sep=';')
    print(f"Saved beta-neutral weights to {output_file}")

    # === OOS signal generation: Generate trading signals for all pairs ===
    if not os.path.exists("signals"):
        os.makedirs("signals")
    signals_list = []
    window = 20         # moving average window
    entry_z = 1.8       # entry threshold
    exit_z = 0.2        # exit threshold
    valid_tickers = set(prices_oos.columns)
    w_df = w_df[w_df['pair'].apply(lambda p: p[0] in valid_tickers and p[1] in valid_tickers)]
    for row in w_df.itertuples():
        x, y = row.pair
        if x not in prices_oos.columns or y not in prices_oos.columns:
            logging.warning(f"Skipping pair {x}-{y} because one of the tickers is missing in OOS price data.")
            continue
        bx, by = row.w_x, row.w_y
        # Spread calculation
        spread = bx * prices_oos[x] + by * prices_oos[y]
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        zscore = (spread - mean) / std
        # Signal generation with position tracking
        signal = []
        position = None  # None, "LONG", "SHORT"
        for i in range(len(zscore)):
            current_z = zscore.iloc[i]
            if pd.isna(current_z):
                signal.append("")
            elif current_z > entry_z:
                if position != "SHORT":
                    signal.append("SELL")
                    position = "SHORT"
                else:
                    signal.append("")
            elif current_z < -entry_z:
                if position != "LONG":
                    signal.append("BUY")
                    position = "LONG"
                else:
                    signal.append("")
            elif abs(current_z) < exit_z:
                if position is not None:
                    signal.append("EXIT")
                    position = None
                else:
                    signal.append("")
            else:
                signal.append("")
        signal = pd.Series(signal, index=spread.index)
        result_df = pd.DataFrame({
            "Date": spread.index,
            "Spread": spread,
            "Z-Score": zscore,
            "Signal": signal
        })
        result_df.to_csv(f"signals/{x}_{y}_signals.csv", sep=";", index=False)
        signals_list.append((x, y))
        print(f"Signals generated for pair: {x}-{y}")
    print(f"\nGenerated signals for {len(signals_list)} pairs.")

    # === First PnL simulation: Simulate OOS trading for all pairs ===
    transaction_cost = 0.001  # 0.1% commission per trade
    signal_folder = "signals"
    results_oos = []
    global_capital = 100000
    max_open_positions = 40
    open_positions = 0
    for fname in os.listdir(signal_folder):
        if not fname.endswith("_signals.csv"):
            continue
        # Skip if position limit reached
        if open_positions >= max_open_positions:
            logging.info(f"Skipping pair {fname.replace('_signals.csv','').replace('_','-')} due to position limit")
            continue
        path = os.path.join(signal_folder, fname)
        df = pd.read_csv(path, sep=';', parse_dates=['Date'])
        x, y = fname.replace("_signals.csv", "").split("_")
        # Check if both tickers are present in OOS price data
        if x not in prices_oos.columns or y not in prices_oos.columns:
            logging.warning(f"Skipping pair {x}-{y}: one of the tickers not in OOS price data.")
            continue
        px = prices_oos[x].loc[df['Date']].values
        py = prices_oos[y].loc[df['Date']].values
        df['Px'] = px
        df['Py'] = py
        pos_x = 0.0
        pos_y = 0.0
        capital = global_capital / max_open_positions
        pnl = []
        position_open = False  # Flag: is position open
        for i in range(len(df)):
            signal = df.loc[i, 'Signal']
            if capital < (global_capital / max_open_positions) * 0.1:
                logging.warning(f"⚠️ Capital critically low: {capital:.2f} at {df.loc[i, 'Date']}")
                break
            if signal == 'BUY':
                if position_open and pos_x < 0:
                    # Close SHORT before entering LONG
                    value_x = pos_x * df.loc[i, 'Px']
                    value_y = pos_y * df.loc[i, 'Py']
                    capital += (value_x + value_y) * (1 - transaction_cost)
                    pos_x = 0
                    pos_y = 0
                    position_open = False
                    open_positions -= 1
                    logging.info(f"AUTO-EXIT before BUY at {df.loc[i, 'Date'].date()} | Capital: {capital:.2f}")
                if not position_open:
                    logging.info(f"Entering LONG at {df.loc[i, 'Date'].date()}")
                    trade_amount = capital * 0.3
                    wx, wy = 1, -1
                    capital *= (1 - transaction_cost)
                    pos_x = (trade_amount / df.loc[i, 'Px']) * wx
                    pos_y = (trade_amount / df.loc[i, 'Py']) * wy
                    position_open = True
                    open_positions += 1
            elif signal == 'SELL':
                if position_open and pos_x > 0:
                    # Close LONG before entering SHORT
                    value_x = pos_x * df.loc[i, 'Px']
                    value_y = pos_y * df.loc[i, 'Py']
                    capital += (value_x + value_y) * (1 - transaction_cost)
                    pos_x = 0
                    pos_y = 0
                    position_open = False
                    open_positions -= 1
                    logging.info(f"AUTO-EXIT before SELL at {df.loc[i, 'Date'].date()} | Capital: {capital:.2f}")
                if not position_open:
                    logging.info(f"Entering SHORT at {df.loc[i, 'Date'].date()}")
                    trade_amount = capital * 0.30
                    wx, wy = -1, 1
                    capital *= (1 - transaction_cost)
                    pos_x = (trade_amount / df.loc[i, 'Px']) * wx
                    pos_y = (trade_amount / df.loc[i, 'Py']) * wy
                    position_open = True
                    open_positions += 1
            elif signal == 'EXIT':
                if position_open:
                    value_x = pos_x * df.loc[i, 'Px']
                    value_y = pos_y * df.loc[i, 'Py']
                    capital += (value_x + value_y) * (1 - transaction_cost)
                    pos_x = 0
                    pos_y = 0
                    position_open = False
                    open_positions -= 1
                    logging.info(f"EXIT at {df.loc[i, 'Date'].date()} | Final capital: {capital:.2f}")
            value = pos_x * df.loc[i, 'Px'] + pos_y * df.loc[i, 'Py']
            total = capital + value
            # --- STOP-LOSS logic ---
            if position_open and total < (global_capital / max_open_positions) * 0.95:
                logging.warning(f"❌ STOP LOSS triggered at {df.loc[i, 'Date']} | Capital: {capital:.2f}")
                capital += (pos_x * df.loc[i, 'Px'] + pos_y * df.loc[i, 'Py']) * (1 - transaction_cost)
                pos_x = 0
                pos_y = 0
                position_open = False
                open_positions -= 1
                logging.info(f"Position forcibly closed due to stop loss.")
            # --- TAKE-PROFIT logic ---
            if position_open and total > (global_capital / max_open_positions) * 1.10:
                logging.info(f"✅ TAKE PROFIT triggered at {df.loc[i, 'Date']} | Capital: {capital:.2f}")
                capital += (pos_x * df.loc[i, 'Px'] + pos_y * df.loc[i, 'Py']) * (1 - transaction_cost)
                pos_x = 0
                pos_y = 0
                position_open = False
                open_positions -= 1
                logging.info(f"Position forcibly closed due to take profit.")
            logging.info(f"{df.loc[i, 'Date'].date()} | Signal: {signal} | Capital: {capital:.2f} | Value: {value:.2f}")
            pnl.append(total)
        df = df.iloc[:len(pnl)]
        df['Portfolio Value'] = pnl
        if not os.path.exists("trading_results"):
            os.makedirs("trading_results")
        df.to_csv(f"trading_results/{x}_{y}_pnl.csv", sep=';', index=False)
        # Final profit
        final_return = (pnl[-1] - (global_capital / max_open_positions)) / (global_capital / max_open_positions)
        sharpe_ratio = calculate_sharpe_ratio(pnl)
        results_oos.append({
            "pair": f"{x}-{y}",
            "final_return": final_return * 100,  # percentage
            "sharpe_ratio": sharpe_ratio
        })

    # === All-pairs OOS summary: Summarize and rank all OOS results ===
    summary_oos = pd.DataFrame(results_oos)
    # Top-N selection
    top_n = 5
    top_pairs_oos_df = summary_oos.sort_values(by='final_return', ascending=False).head(top_n)
    top_pairs_oos = top_pairs_oos_df['pair'].tolist()
    summary_oos['final_return'] = summary_oos['final_return'].round(2)
    summary_oos['sharpe_ratio'] = summary_oos['sharpe_ratio'].round(3)
    if not os.path.exists("trading_results"):
        os.makedirs("trading_results")
    summary_oos['final_profit'] = summary_oos['final_return'].map(lambda x: round(x/100 * (global_capital / max_open_positions), 2))
    summary_oos['final_return_pct'] = summary_oos['final_return'].map(lambda x: f"{x:.2f}%")
    summary_oos_to_save = summary_oos[['pair', 'final_return_pct', 'final_profit', 'sharpe_ratio']].rename(columns={
        'final_return_pct': 'final_return (%)',
        'final_profit': 'final_profit ($)',
        'sharpe_ratio': 'sharpe_ratio'
    })
    summary_oos_to_save.to_csv("trading_results/summary_returns.csv", sep=';', index=False)
    print("Saved all PnL results.")
    mean_return = summary_oos['final_return'].mean()
    median_return = summary_oos['final_return'].median()
    std_return = summary_oos['final_return'].std()
    mean_sharpe = summary_oos['sharpe_ratio'].mean()
    median_sharpe = summary_oos['sharpe_ratio'].median()
    profitable = (summary_oos['final_return'] > 0).sum()
    unprofitable = (summary_oos['final_return'] <= 0).sum()
    total_pairs = len(summary_oos)
    logging.info(f"\n=== In-Sample / Out-of-Sample Setup ===")
    logging.info(f"In-sample period: {is_start} to {is_end}")
    logging.info(f"Out-of-sample period: {oos_start} to {oos_end}")
    logging.info("Pairs, betas, and beta-neutral weights were estimated on IS data only.")
    logging.info("Trading signals and PnL simulation were evaluated on OOS data only.")
    logging.info(f"\n=== Summary Statistics ===")
    logging.info(f"Total pairs tested: {total_pairs}")
    logging.info(f"Profitable pairs: {profitable}")
    logging.info(f"Unprofitable pairs: {unprofitable}")
    logging.info(f"Average return: {mean_return:.2f}% (${mean_return / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Median return: {median_return:.2f}% (${median_return / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Standard deviation of return: {std_return:.2f}% (${std_return / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Average Sharpe ratio: {mean_sharpe:.3f}")
    logging.info(f"Median Sharpe ratio: {median_sharpe:.3f}")
    # Additional metrics
    max_return = summary_oos['final_return'].max()
    min_return = summary_oos['final_return'].min()
    win_rate = profitable / total_pairs if total_pairs > 0 else 0
    loss_rate = unprofitable / total_pairs if total_pairs > 0 else 0
    logging.info(f"Max return: {max_return:.2f}% (${max_return / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Min return: {min_return:.2f}% (${min_return / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Win rate: {win_rate:.2%}")
    logging.info(f"Loss rate: {loss_rate:.2%}")
    total_profit = summary_oos['final_return'].sum() / 100 * global_capital / max_open_positions
    logging.info(f"Total profit from all trades: ${total_profit:.2f}")
    logging.info(f"Simulating again using top {top_n} pairs for refined strategy...")

    # === Second top-N simulation: Simulate OOS trading for top-N pairs ===
    results_top_oos = []
    open_positions = 0
    for fname in os.listdir(signal_folder):
        if not fname.endswith("_signals.csv"):
            continue
        pair_name = fname.replace("_signals.csv", "").replace("_", "-")
        if pair_name not in top_pairs_oos:
            continue
        path = os.path.join(signal_folder, fname)
        df = pd.read_csv(path, sep=';', parse_dates=['Date'])
        x, y = fname.replace("_signals.csv", "").split("_")
        # Check if both tickers are present in OOS price data
        if x not in prices_oos.columns or y not in prices_oos.columns:
            logging.warning(f"Skipping pair {x}-{y}: one of the tickers not in OOS price data.")
            continue
        px = prices_oos[x].loc[df['Date']].values
        py = prices_oos[y].loc[df['Date']].values
        df['Px'] = px
        df['Py'] = py
        pos_x = 0.0
        pos_y = 0.0
        capital = global_capital / max_open_positions
        pnl = []
        position_open = False  # Flag: is position open
        for i in range(len(df)):
            signal = df.loc[i, 'Signal']
            if capital < (global_capital / max_open_positions) * 0.1:
                logging.warning(f"⚠️ Capital critically low: {capital:.2f} at {df.loc[i, 'Date']}")
                break
            if signal == 'BUY':
                if position_open and pos_x < 0:
                    # Close SHORT before entering LONG
                    value_x = pos_x * df.loc[i, 'Px']
                    value_y = pos_y * df.loc[i, 'Py']
                    capital += (value_x + value_y) * (1 - transaction_cost)
                    pos_x = 0
                    pos_y = 0
                    position_open = False
                    open_positions -= 1
                    logging.info(f"AUTO-EXIT before BUY at {df.loc[i, 'Date'].date()} | Capital: {capital:.2f}")
                if not position_open:
                    logging.info(f"Entering LONG at {df.loc[i, 'Date'].date()}")
                    trade_amount = capital * 0.30
                    wx, wy = 1, -1
                    capital *= (1 - transaction_cost)
                    pos_x = (trade_amount / df.loc[i, 'Px']) * wx
                    pos_y = (trade_amount / df.loc[i, 'Py']) * wy
                    position_open = True
                    open_positions += 1
            elif signal == 'SELL':
                if position_open and pos_x > 0:
                    # Close LONG before entering SHORT
                    value_x = pos_x * df.loc[i, 'Px']
                    value_y = pos_y * df.loc[i, 'Py']
                    capital += (value_x + value_y) * (1 - transaction_cost)
                    pos_x = 0
                    pos_y = 0
                    position_open = False
                    open_positions -= 1
                    logging.info(f"AUTO-EXIT before SELL at {df.loc[i, 'Date'].date()} | Capital: {capital:.2f}")
                if not position_open:
                    logging.info(f"Entering SHORT at {df.loc[i, 'Date'].date()}")
                    trade_amount = capital * 0.30
                    wx, wy = -1, 1
                    capital *= (1 - transaction_cost)
                    pos_x = (trade_amount / df.loc[i, 'Px']) * wx
                    pos_y = (trade_amount / df.loc[i, 'Py']) * wy
                    position_open = True
                    open_positions += 1
            elif signal == 'EXIT':
                if position_open:
                    value_x = pos_x * df.loc[i, 'Px']
                    value_y = pos_y * df.loc[i, 'Py']
                    capital += (value_x + value_y) * (1 - transaction_cost)
                    pos_x = 0
                    pos_y = 0
                    position_open = False
                    open_positions -= 1
                    logging.info(f"EXIT at {df.loc[i, 'Date'].date()} | Final capital: {capital:.2f}")
            value = pos_x * df.loc[i, 'Px'] + pos_y * df.loc[i, 'Py']
            total = capital + value
            # --- STOP-LOSS logic ---
            if position_open and total < (global_capital / max_open_positions) * 0.95:
                logging.warning(f"❌ STOP LOSS triggered at {df.loc[i, 'Date']} | Capital: {capital:.2f}")
                capital += (pos_x * df.loc[i, 'Px'] + pos_y * df.loc[i, 'Py']) * (1 - transaction_cost)
                pos_x = 0
                pos_y = 0
                position_open = False
                open_positions -= 1
                logging.info(f"Position forcibly closed due to stop loss.")
            # --- TAKE-PROFIT logic ---
            if position_open and total > (global_capital / max_open_positions) * 1.10:
                logging.info(f"✅ TAKE PROFIT triggered at {df.loc[i, 'Date']} | Capital: {capital:.2f}")
                capital += (pos_x * df.loc[i, 'Px'] + pos_y * df.loc[i, 'Py']) * (1 - transaction_cost)
                pos_x = 0
                pos_y = 0
                position_open = False
                open_positions -= 1
                logging.info(f"Position forcibly closed due to take profit.")
            logging.info(f"{df.loc[i, 'Date'].date()} | Signal: {signal} | Capital: {capital:.2f} | Value: {value:.2f}")
            pnl.append(total)
        df = df.iloc[:len(pnl)]
        df['Portfolio Value'] = pnl
        if not os.path.exists("trading_results"):
            os.makedirs("trading_results")
        df.to_csv(f"trading_results/{x}_{y}_pnl.csv", sep=';', index=False)
        # Final profit
        final_return = (pnl[-1] - (global_capital / max_open_positions)) / (global_capital / max_open_positions)
        sharpe_ratio = calculate_sharpe_ratio(pnl)
        results_top_oos.append({
            "pair": f"{x}-{y}",
            "final_return": final_return * 100,  # percentage
            "sharpe_ratio": sharpe_ratio
        })

    # === Top-N summary export: Export top-N simulation results ===
    summary_top_oos = pd.DataFrame(results_top_oos)
    summary_top_oos['final_return'] = summary_top_oos['final_return'].round(2)
    summary_top_oos['sharpe_ratio'] = summary_top_oos['sharpe_ratio'].round(3)
    summary_top_oos['final_profit'] = summary_top_oos['final_return'].map(lambda x: round(x/100 * (global_capital / max_open_positions), 2))
    summary_top_oos['final_return_pct'] = summary_top_oos['final_return'].map(lambda x: f"{x:.2f}%")
    summary_top_oos_to_save = summary_top_oos[['pair', 'final_return_pct', 'final_profit', 'sharpe_ratio']].rename(columns={
        'final_return_pct': 'final_return (%)',
        'final_profit': 'final_profit ($)',
        'sharpe_ratio': 'sharpe_ratio'
    })
    summary_top_oos_to_save.to_csv("trading_results/summary_returns_top.csv", sep=';', index=False)
    print("Saved top N PnL results to trading_results/summary_returns_top.csv")

    # --- Top-N summary statistics ---
    mean_return_top = summary_top_oos['final_return'].mean()
    median_return_top = summary_top_oos['final_return'].median()
    std_return_top = summary_top_oos['final_return'].std()
    mean_sharpe_top = summary_top_oos['sharpe_ratio'].mean()
    median_sharpe_top = summary_top_oos['sharpe_ratio'].median()
    profitable_top = (summary_top_oos['final_return'] > 0).sum()
    unprofitable_top = (summary_top_oos['final_return'] <= 0).sum()
    total_pairs_top = len(summary_top_oos)
    max_return_top = summary_top_oos['final_return'].max()
    min_return_top = summary_top_oos['final_return'].min()
    win_rate_top = profitable_top / total_pairs_top if total_pairs_top > 0 else 0
    loss_rate_top = unprofitable_top / total_pairs_top if total_pairs_top > 0 else 0
    total_profit_top = summary_top_oos['final_return'].sum() / 100 * global_capital / max_open_positions
    logging.info(f"\n=== Summary Statistics for Top {top_n} Pairs ===")
    logging.info(f"Total pairs tested: {total_pairs_top}")
    logging.info(f"Profitable pairs: {profitable_top}")
    logging.info(f"Unprofitable pairs: {unprofitable_top}")
    logging.info(f"Average return: {mean_return_top:.2f}% (${mean_return_top / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Median return: {median_return_top:.2f}% (${median_return_top / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Standard deviation of return: {std_return_top:.2f}% (${std_return_top / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Average Sharpe ratio: {mean_sharpe_top:.3f}")
    logging.info(f"Median Sharpe ratio: {median_sharpe_top:.3f}")
    logging.info(f"Max return: {max_return_top:.2f}% (${max_return_top / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Min return: {min_return_top:.2f}% (${min_return_top / 100 * global_capital / max_open_positions:.2f})")
    logging.info(f"Win rate: {win_rate_top:.2%}")
    logging.info(f"Loss rate: {loss_rate_top:.2%}")
    logging.info(f"Total profit from top trades: ${total_profit_top:.2f}")
