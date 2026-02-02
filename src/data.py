import pandas as pd
import yfinance as yf
from loguru import logger


def load_prices(tickers, start="2020-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches stock data (Prices and Volume).

    Args:
        tickers: List of ticker symbols.
        start: Start date string (YYYY-MM-DD).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Adj Close Prices, Volume)
    """
    data = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError("No data fetched. Please check the tickers and date range.")

    prices = data["Close"].ffill().bfill()
    volume = (
        data["Volume"].ffill().bfill()
        if "Volume" in data
        else pd.DataFrame(0, index=prices.index, columns=prices.columns)
    )

    return prices, volume


def fill_missing_dates(prices: pd.DataFrame, method: str = "linear", fill_weekends: bool = True) -> pd.DataFrame:
    """Reindexes prices to include all calendar days (or business days) and interpolates missing values.

    Args:
        prices: DataFrame with DateTime index.
        method: Interpolation method ('linear', 'ffill', etc.). Defaults to 'linear'.
        fill_weekends: If True, reindexes to include all calendar days (freq='D');
                       otherwise, reindexes only to business days (freq='B').

    Returns:
        pd.DataFrame: DataFrame with daily or business day frequency and interpolated values.
    """
    if prices.empty:
        return prices

    freq = "D" if fill_weekends else "B"
    all_days = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq=freq)
    prices = prices.reindex(all_days)

    if method == "linear":
        prices = prices.interpolate(method="linear")
    elif method == "ffill":
        prices = prices.ffill()
    else:
        raise ValueError(f"Unknown interpolation method: {method}. Supported methods are 'linear' and 'ffill'.")

    return prices


def smooth_prices(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Applies moving average smoothing to price data for each column (asset).

    Args:
        prices (pd.DataFrame): DataFrame with DateTime index and asset prices.
        window (int, optional): Window size for moving average. Defaults to 5.
    Returns:
        pd.DataFrame: Smoothed price DataFrame.
    """
    smoothed = prices.rolling(window=window, min_periods=1).mean()
    return smoothed


def process_returns(
    prices_df: pd.DataFrame,
    delta_t: int = 365,
    apply_smoothing: bool = False,
    smoothing_window: int = 5,
    fill_missing: bool = False,
    fill_weekends: bool = True,
    volume_df: pd.DataFrame | None = None,
    min_liquidity: float = 0.0,
) -> dict:
    """Calculates returns and cleans data for analysis.

    Args:
        prices_df: DataFrame of asset prices.
        delta_t: Number of days to look back for analysis.
        apply_smoothing: Whether to apply moving average smoothing.
        smoothing_window: Window size for smoothing.
        fill_missing: Whether to interpolate missing calendar days (weekends/holidays).
        volume_df: DataFrame of asset volumes (optional, for liquidity filtering).
        min_liquidity: Minimum average daily turnover (Price * Volume) to consider a stock valid.

    Returns:
        dict: Dictionary containing returns statistics and historical data.
    """
    if fill_missing:
        prices_df = fill_missing_dates(prices_df, fill_weekends=fill_weekends)
        if volume_df is not None:
            volume_df = fill_missing_dates(
                volume_df, method="ffill", fill_weekends=fill_weekends
            )

    if apply_smoothing:
        prices_df = smooth_prices(prices_df, window=smoothing_window)

    if volume_df is not None and min_liquidity > 0:
        common_idx = prices_df.index.intersection(volume_df.index)
        period_start = common_idx[-delta_t - 1] if len(common_idx) > delta_t else common_idx[0]

        p_slice = prices_df.loc[period_start:]
        v_slice = volume_df.loc[period_start:]

        daily_turnover = p_slice * v_slice
        avg_turnover = daily_turnover.mean()

        valid_liquidity_cols = avg_turnover[avg_turnover >= min_liquidity].index

        prices_df = prices_df[valid_liquidity_cols]

    stock_returns = prices_df.pct_change(fill_method=None)
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


def load_benchmark(ticker: str, start: str) -> pd.Series | None:
    """Loads benchmark data, handling potential errors.

    Args:
        ticker: Benchmark ticker symbol.
        start: Start date string (YYYY-MM-DD).

    Returns:
        pd.Series or None: Benchmark price series or None if failed.
    """
    if not ticker:
        return None
    try:
        benchmark_data, _ = load_prices([ticker], start=start)
        if isinstance(benchmark_data, pd.DataFrame) and not benchmark_data.empty:
            benchmark = benchmark_data.iloc[:, 0]
        else:
            logger.warning(f"[WARN] No valid price data fetched for benchmark {ticker}.")
            return None

        valid_count = benchmark.notna().sum()
        if valid_count < 50:
            logger.warning(f"[WARN] Benchmark {ticker} has only {valid_count} valid prices - skipping")
            return None
        return benchmark
    except Exception as exc:
        logger.warning(f"Could not load benchmark {ticker}: {exc}")
        return None


def create_synthetic_index(prices_df: pd.DataFrame, method: str = "equal") -> pd.Series:
    """Create a synthetic index from component stocks.

    Args:
        prices_df: DataFrame of stock prices (columns are tickers).
        method: 'equal' for equal-weight, 'price' for price-weighted.

    Returns:
        pd.Series: Synthetic index price series.
    """
    if prices_df.empty:
        return pd.Series(dtype=float)

    if method == "equal":
        returns = prices_df.pct_change(fill_method=None)
        avg_returns = returns.mean(axis=1, skipna=True).fillna(0)
        synthetic = 100 * (1 + avg_returns).cumprod()

    elif method == "price":
        synthetic = prices_df.sum(axis=1)
        if not synthetic.empty:
            synthetic = 100 * synthetic / synthetic.iloc[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    synthetic.name = "Synthetic Index"
    return synthetic
