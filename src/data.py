import pandas as pd
import yfinance as yf


def load_prices(tickers, start="2020-01-01") -> pd.Series:
    """Fetches WIG20 data. Uses Adjusted Close for returns as it accounts for dividends and splits."""
    data = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError("No data fetched. Please check the tickers and date range.")

    prices = data["Close"].ffill().bfill()
    return prices


def load_raw_data(tickers, start="2020-01-01"):
    """Fetches full ticker data from Yahoo Finance."""
    data = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError("No data fetched. Please check the tickers and date range.")
    return data


def process_returns(prices_df: pd.DataFrame, delta_t=365) -> dict:
    """Calculates returns and cleans data for analysis."""
    stock_returns = prices_df.pct_change()
    analysis_period = stock_returns.iloc[-delta_t - 1 : -1]

    valid_stocks_mask = ~analysis_period.isna().any()
    valid_stocks = analysis_period.columns[valid_stocks_mask]
    analysis_period_clean = analysis_period[valid_stocks]

    stock_returns_m = analysis_period_clean.mean()
    stock_returns_s = analysis_period_clean.std()
    stock_covariances = analysis_period_clean.cov()
    stock_correlations = analysis_period_clean.corr()

    return {
        "returns_m": stock_returns_m,
        "returns_s": stock_returns_s,
        "covariances": stock_covariances,
        "correlations": stock_correlations,
        "valid_tickers": valid_stocks.tolist(),
        "historical_returns": analysis_period_clean,
    }


def interpolate_weekends(prices: pd.Series) -> pd.Series:
    """Interpolates missing weekend data in price DataFrame using linear interpolation.

    Args:
        prices (pd.Series): Series with DateTime index and asset prices
    Returns:
        pd.DataFrame: DataFrame with weekends interpolated"""
    all_days = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq="D")
    prices = prices.reindex(all_days)
    prices = prices.interpolate(method="linear")
    return prices


def smooth_prices(prices: pd.Series, window: int = 5) -> pd.Series:
    """Applies moving average smoothing to price data.

    Args:
        prices (pd.Series): Series with DateTime index and asset prices
        window (int, optional): Window size for moving average. Defaults to 5.
    Returns:
        pd.Series: Smoothed price Series
    """
    smoothed = prices.rolling(window=window, min_periods=1).mean()
    return smoothed
