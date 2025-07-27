import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class QuantitativeFactorModel:
    """
    A comprehensive factor model implementation using real market data from Yahoo Finance
    Includes Fama-French factors and extensions with actual market data
    """

    def __init__(self):
        self.factors = None
        self.returns = None
        self.results = {}
        self.factor_loadings = None
        self.stock_data = None

    def fetch_market_data(self, tickers=None, start_date="2020-01-01", end_date=None):
        """
        Fetch real market data from Yahoo Finance
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if tickers is None:
            # Default portfolio of diverse stocks
            tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech/Growth
                'JPM', 'BAC', 'WFC', 'GS', 'MS',  # Finance
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',  # Healthcare
                'XOM', 'CVX', 'COP', 'SLB', 'EOG',  # Energy
                'WMT', 'PG', 'KO', 'PEP', 'MCD'  # Consumer
            ]

        print(f"Fetching data for {len(tickers)} stocks from {start_date} to {end_date}...")

        try:
            # Download stock data
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

            # Handle MultiIndex columns (when multiple tickers) vs single level (single ticker)
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Multiple tickers - extract Adj Close
                if 'Adj Close' in raw_data.columns.get_level_values(0):
                    stock_data = raw_data['Adj Close']
                else:
                    # Fallback to Close if Adj Close not available
                    stock_data = raw_data['Close']
            else:
                # Single ticker case - data comes as single level columns
                if len(tickers) == 1:
                    # Create a DataFrame with ticker as column name
                    if 'Adj Close' in raw_data.columns:
                        stock_data = raw_data[['Adj Close']].copy()
                        stock_data.columns = tickers
                    else:
                        stock_data = raw_data[['Close']].copy()
                        stock_data.columns = tickers
                else:
                    # Multiple tickers but got single level - shouldn't happen but handle it
                    stock_data = raw_data

            # Ensure we have a DataFrame
            if isinstance(stock_data, pd.Series):
                stock_data = stock_data.to_frame(tickers[0])

            # Remove any columns that are entirely NaN
            stock_data = stock_data.dropna(axis=1, how='all')

            # Calculate returns
            stock_returns = stock_data.pct_change().dropna()

            # Remove any stocks with insufficient data
            min_observations = max(50, len(stock_returns) * 0.5)  # At least 50 obs or 50% of data
            valid_stocks = stock_returns.count() >= min_observations
            stock_returns = stock_returns.loc[:, valid_stocks]
            stock_data = stock_data.loc[:, valid_stocks]

            print(f"Successfully loaded {len(stock_returns.columns)} stocks with {len(stock_returns)} observations")
            print(f"Valid tickers: {list(stock_returns.columns)}")

            self.returns = stock_returns
            self.stock_data = stock_data
            return stock_returns

        except Exception as e:
            print(f"Error fetching market data: {e}")
            print("Trying individual ticker approach...")

            # Fallback: fetch tickers individually
            stock_data_dict = {}
            for ticker in tickers:
                try:
                    ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not ticker_data.empty:
                        if 'Adj Close' in ticker_data.columns:
                            stock_data_dict[ticker] = ticker_data['Adj Close']
                        else:
                            stock_data_dict[ticker] = ticker_data['Close']
                except Exception as tick_err:
                    print(f"Failed to fetch {ticker}: {tick_err}")
                    continue

            if stock_data_dict:
                stock_data = pd.DataFrame(stock_data_dict)
                stock_returns = stock_data.pct_change().dropna()

                # Remove stocks with insufficient data
                min_observations = max(50, len(stock_returns) * 0.5)
                valid_stocks = stock_returns.count() >= min_observations
                stock_returns = stock_returns.loc[:, valid_stocks]
                stock_data = stock_data.loc[:, valid_stocks]

                print(f"Fallback successful: {len(stock_returns.columns)} stocks loaded")

                self.returns = stock_returns
                self.stock_data = stock_data
                return stock_returns
            else:
                print("Failed to fetch any stock data")
                return None

    def create_factors_from_market_data(self):
        """
        Create factor returns from market data using proxies
        """
        if self.returns is None:
            print("Fetch market data first!")
            return

        print("Creating factor proxies from market data...")

        # Get additional market data for factors
        start_date = self.returns.index[0].strftime("%Y-%m-%d")
        end_date = self.returns.index[-1].strftime("%Y-%m-%d")

        # Market factor - use SPY as market proxy
        print("Fetching SPY data...")
        try:
            spy_raw = yf.download('SPY', start=start_date, end=end_date, progress=False)
            if isinstance(spy_raw.columns, pd.MultiIndex):
                market_data = spy_raw['Adj Close']['SPY'] if 'SPY' in spy_raw['Adj Close'].columns else spy_raw[
                                                                                                            'Adj Close'].iloc[
                                                                                                        :, 0]
            else:
                market_data = spy_raw['Adj Close'] if 'Adj Close' in spy_raw.columns else spy_raw['Close']
            market_returns = market_data.pct_change().dropna()
        except Exception as e:
            print(f"SPY fetch failed: {e}, using stock universe mean as market proxy")
            market_returns = self.returns.mean(axis=1)

        # Risk-free rate - use 3-month Treasury (^IRX)
        try:
            print("Fetching Treasury rate data...")
            rf_raw = yf.download('^IRX', start=start_date, end=end_date, progress=False)
            if isinstance(rf_raw.columns, pd.MultiIndex):
                rf_data = rf_raw['Adj Close'].iloc[:, 0] if not rf_raw.empty else None
            else:
                rf_data = rf_raw['Adj Close'] if 'Adj Close' in rf_raw.columns else rf_raw['Close']

            if rf_data is not None:
                rf_returns = (rf_data / 100 / 252).reindex(market_returns.index).fillna(method='ffill').fillna(
                    0.02 / 252)
            else:
                raise ValueError("No RF data available")
        except Exception as e:
            print(f"Treasury data unavailable, using constant rate: {e}")
            rf_returns = pd.Series(0.02 / 252, index=market_returns.index)  # 2% annual

        # Market excess return
        market_excess = market_returns - rf_returns

        # Initialize common_dates with market data
        common_dates = market_returns.index.intersection(self.returns.index)

        # Size factor (SMB) - Small cap vs Large cap proxy
        try:
            print("Fetching size factor ETF data...")
            iwm_raw = yf.download('IWM', start=start_date, end=end_date, progress=False)
            iwb_raw = yf.download('IWB', start=start_date, end=end_date, progress=False)

            # Handle MultiIndex or single level columns
            if isinstance(iwm_raw.columns, pd.MultiIndex):
                iwm_data = iwm_raw['Adj Close'].iloc[:, 0] if not iwm_raw.empty else None
            else:
                iwm_data = iwm_raw['Adj Close'] if 'Adj Close' in iwm_raw.columns else iwm_raw['Close']

            if isinstance(iwb_raw.columns, pd.MultiIndex):
                iwb_data = iwb_raw['Adj Close'].iloc[:, 0] if not iwb_raw.empty else None
            else:
                iwb_data = iwb_raw['Adj Close'] if 'Adj Close' in iwb_raw.columns else iwb_raw['Close']

            if iwm_data is not None and iwb_data is not None:
                iwm_returns = iwm_data.pct_change().dropna()
                iwb_returns = iwb_data.pct_change().dropna()

                # Align dates and create SMB factor
                etf_common = iwm_returns.index.intersection(iwb_returns.index).intersection(common_dates)
                if len(etf_common) > 50:  # Need sufficient data
                    smb_factor = pd.Series(
                        iwm_returns.loc[etf_common].values - iwb_returns.loc[etf_common].values,
                        index=etf_common,
                        name='SMB'
                    )
                    common_dates = etf_common
                else:
                    raise ValueError("Insufficient ETF data")
            else:
                raise ValueError("Could not fetch ETF data")

        except Exception as e:
            print(f"ETF data issue, creating SMB from stock universe: {e}")
            # Fallback: create SMB from stock universe
            try:
                avg_prices = self.stock_data.tail(20).mean()
                small_cap_stocks = avg_prices.nsmallest(len(avg_prices) // 2).index
                large_cap_stocks = avg_prices.nlargest(len(avg_prices) // 2).index

                small_cap_ret = self.returns[small_cap_stocks].mean(axis=1)
                large_cap_ret = self.returns[large_cap_stocks].mean(axis=1)
                smb_factor = (small_cap_ret - large_cap_ret).dropna()
                smb_factor.name = 'SMB'
                common_dates = common_dates.intersection(smb_factor.index)
            except Exception as e2:
                print(f"Fallback SMB creation failed: {e2}")
                # Ultimate fallback - create random walk SMB
                smb_factor = pd.Series(
                    np.random.normal(0, 0.01, len(common_dates)),
                    index=common_dates,
                    name='SMB'
                )

        # Value factor (HML) - Value vs Growth proxy
        try:
            print("Fetching value factor ETF data...")
            iwd_raw = yf.download('IWD', start=start_date, end=end_date, progress=False)
            iwf_raw = yf.download('IWF', start=start_date, end=end_date, progress=False)

            # Handle MultiIndex or single level columns
            if isinstance(iwd_raw.columns, pd.MultiIndex):
                iwd_data = iwd_raw['Adj Close'].iloc[:, 0] if not iwd_raw.empty else None
            else:
                iwd_data = iwd_raw['Adj Close'] if 'Adj Close' in iwd_raw.columns else iwd_raw['Close']

            if isinstance(iwf_raw.columns, pd.MultiIndex):
                iwf_data = iwf_raw['Adj Close'].iloc[:, 0] if not iwf_raw.empty else None
            else:
                iwf_data = iwf_raw['Adj Close'] if 'Adj Close' in iwf_raw.columns else iwf_raw['Close']

            if iwd_data is not None and iwf_data is not None:
                iwd_returns = iwd_data.pct_change().dropna()
                iwf_returns = iwf_data.pct_change().dropna()

                value_common = iwd_returns.index.intersection(iwf_returns.index).intersection(common_dates)
                if len(value_common) > 50:
                    hml_factor = pd.Series(
                        iwd_returns.loc[value_common].values - iwf_returns.loc[value_common].values,
                        index=value_common,
                        name='HML'
                    )
                    common_dates = common_dates.intersection(value_common)
                else:
                    raise ValueError("Insufficient value ETF data")
            else:
                raise ValueError("Could not fetch value ETF data")

        except Exception as e:
            print(f"Value ETF issue, creating HML from stock data: {e}")
            # Fallback: create simple mean reversion factor
            try:
                returns_12m = self.returns.rolling(min_periods=60, window=252).sum()
                returns_12m = returns_12m.dropna()

                # Create value/growth portfolios based on past returns (contrarian)
                value_portfolio = []
                growth_portfolio = []

                for date in returns_12m.index:
                    if date in common_dates:
                        day_returns = returns_12m.loc[date].dropna()
                        if len(day_returns) >= 6:  # Need minimum stocks
                            bottom_30 = day_returns.nsmallest(max(1, len(day_returns) // 3))
                            top_30 = day_returns.nlargest(max(1, len(day_returns) // 3))

                            if date in self.returns.index:
                                value_ret = self.returns.loc[date, bottom_30.index].mean()
                                growth_ret = self.returns.loc[date, top_30.index].mean()
                                value_portfolio.append(value_ret)
                                growth_portfolio.append(growth_ret)
                            else:
                                value_portfolio.append(0)
                                growth_portfolio.append(0)
                        else:
                            value_portfolio.append(0)
                            growth_portfolio.append(0)

                if len(value_portfolio) > 0:
                    hml_factor = pd.Series(
                        np.array(value_portfolio) - np.array(growth_portfolio),
                        index=returns_12m.index.intersection(common_dates),
                        name='HML'
                    ).dropna()
                    common_dates = common_dates.intersection(hml_factor.index)
                else:
                    raise ValueError("Could not create HML factor")

            except Exception as e2:
                print(f"HML fallback failed: {e2}")
                hml_factor = pd.Series(
                    np.random.normal(0, 0.008, len(common_dates)),
                    index=common_dates,
                    name='HML'
                )

        # Momentum factor
        try:
            print("Fetching momentum factor data...")
            mtum_data = yf.download('MTUM', start=start_date, end=end_date, progress=False)['Adj Close']
            mtum_returns = mtum_data.pct_change().dropna()

            mom_common = mtum_returns.index.intersection(common_dates)
            if len(mom_common) > 50:
                # Use MTUM excess return over market
                mom_factor = pd.Series(
                    mtum_returns.loc[mom_common].values - market_returns.loc[mom_common].values,
                    index=mom_common,
                    name='MOM'
                )
                common_dates = common_dates.intersection(mom_common)
            else:
                raise ValueError("Insufficient momentum ETF data")

        except Exception as e:
            print(f"Momentum ETF issue, creating from stock data: {e}")
            try:
                # Create momentum factor from stock universe (12-1 month returns)
                returns_12m = self.returns.rolling(min_periods=60, window=252).sum().shift(21)
                returns_12m = returns_12m.dropna()

                momentum_portfolio = []
                weak_portfolio = []

                for date in returns_12m.index:
                    if date in common_dates:
                        day_mom = returns_12m.loc[date].dropna()
                        if len(day_mom) >= 6:
                            top_30 = day_mom.nlargest(max(1, len(day_mom) // 3))
                            bottom_30 = day_mom.nsmallest(max(1, len(day_mom) // 3))

                            if date in self.returns.index:
                                mom_ret = self.returns.loc[date, top_30.index].mean()
                                weak_ret = self.returns.loc[date, bottom_30.index].mean()
                                momentum_portfolio.append(mom_ret)
                                weak_portfolio.append(weak_ret)
                            else:
                                momentum_portfolio.append(0)
                                weak_portfolio.append(0)
                        else:
                            momentum_portfolio.append(0)
                            weak_portfolio.append(0)

                if len(momentum_portfolio) > 0:
                    mom_factor = pd.Series(
                        np.array(momentum_portfolio) - np.array(weak_portfolio),
                        index=returns_12m.index.intersection(common_dates),
                        name='MOM'
                    ).dropna()
                    common_dates = common_dates.intersection(mom_factor.index)
                else:
                    raise ValueError("Could not create momentum factor")

            except Exception as e2:
                print(f"Momentum fallback failed: {e2}")
                mom_factor = pd.Series(
                    np.random.normal(0, 0.012, len(common_dates)),
                    index=common_dates,
                    name='MOM'
                )

        # Quality factor - use QUAL ETF proxy
        try:
            print("Fetching quality factor data...")
            qual_data = yf.download('QUAL', start=start_date, end=end_date, progress=False)['Adj Close']
            qual_returns = qual_data.pct_change().dropna()

            qual_common = qual_returns.index.intersection(common_dates)
            if len(qual_common) > 50:
                quality_factor = pd.Series(
                    qual_returns.loc[qual_common].values - market_returns.loc[qual_common].values,
                    index=qual_common,
                    name='QMJ'
                )
                common_dates = common_dates.intersection(qual_common)
            else:
                raise ValueError("Insufficient quality ETF data")

        except Exception as e:
            print(f"Quality ETF issue, using low volatility proxy: {e}")
            # Fallback: use low volatility as quality proxy
            try:
                vol_60d = self.returns.rolling(window=60, min_periods=30).std()
                vol_60d = vol_60d.dropna()

                quality_portfolio = []
                junk_portfolio = []

                for date in vol_60d.index:
                    if date in common_dates:
                        day_vol = vol_60d.loc[date].dropna()
                        if len(day_vol) >= 6:
                            low_vol = day_vol.nsmallest(max(1, len(day_vol) // 3))
                            high_vol = day_vol.nlargest(max(1, len(day_vol) // 3))

                            if date in self.returns.index:
                                qual_ret = self.returns.loc[date, low_vol.index].mean()
                                junk_ret = self.returns.loc[date, high_vol.index].mean()
                                quality_portfolio.append(qual_ret)
                                junk_portfolio.append(junk_ret)
                            else:
                                quality_portfolio.append(0)
                                junk_portfolio.append(0)
                        else:
                            quality_portfolio.append(0)
                            junk_portfolio.append(0)

                if len(quality_portfolio) > 0:
                    quality_factor = pd.Series(
                        np.array(quality_portfolio) - np.array(junk_portfolio),
                        index=vol_60d.index.intersection(common_dates),
                        name='QMJ'
                    ).dropna()
                    common_dates = common_dates.intersection(quality_factor.index)
                else:
                    raise ValueError("Could not create quality factor")

            except Exception as e2:
                print(f"Quality fallback failed: {e2}")
                quality_factor = pd.Series(
                    np.random.normal(0, 0.007, len(common_dates)),
                    index=common_dates,
                    name='QMJ'
                )

        # Low volatility factor
        try:
            print("Creating low volatility factor...")
            vol_20d = self.returns.rolling(window=20, min_periods=10).std()
            vol_20d = vol_20d.dropna()

            lv_portfolio = []
            hv_portfolio = []

            for date in vol_20d.index:
                if date in common_dates:
                    day_vol = vol_20d.loc[date].dropna()
                    if len(day_vol) >= 6:
                        low_vol = day_vol.nsmallest(max(1, len(day_vol) // 3))
                        high_vol = day_vol.nlargest(max(1, len(day_vol) // 3))

                        if date in self.returns.index:
                            lv_ret = self.returns.loc[date, low_vol.index].mean()
                            hv_ret = self.returns.loc[date, high_vol.index].mean()
                            lv_portfolio.append(lv_ret)
                            hv_portfolio.append(hv_ret)
                        else:
                            lv_portfolio.append(0)
                            hv_portfolio.append(0)
                    else:
                        lv_portfolio.append(0)
                        hv_portfolio.append(0)

            if len(lv_portfolio) > 0:
                lv_factor = pd.Series(
                    np.array(lv_portfolio) - np.array(hv_portfolio),
                    index=vol_20d.index.intersection(common_dates),
                    name='LV'
                ).dropna()
                common_dates = common_dates.intersection(lv_factor.index)
            else:
                raise ValueError("Could not create low vol factor")

        except Exception as e:
            print(f"Low vol factor creation failed: {e}")
            lv_factor = pd.Series(
                np.random.normal(0, 0.005, len(common_dates)),
                index=common_dates,
                name='LV'
            )

        # Ensure all factors have the same final index
        final_dates = common_dates
        print(f"Final alignment: {len(final_dates)} common dates")

        if len(final_dates) < 50:
            print("Warning: Very few common dates found. Check data availability.")

        # Create factor dataframe with proper alignment
        try:
            factor_dict = {
                'Mkt-RF': market_excess.reindex(final_dates).fillna(0),
                'SMB': smb_factor.reindex(final_dates).fillna(0),
                'HML': hml_factor.reindex(final_dates).fillna(0),
                'MOM': mom_factor.reindex(final_dates).fillna(0),
                'QMJ': quality_factor.reindex(final_dates).fillna(0),
                'LV': lv_factor.reindex(final_dates).fillna(0),
                'RF': rf_returns.reindex(final_dates).fillna(0.02 / 252)
            }

            self.factors = pd.DataFrame(factor_dict, index=final_dates)

            # Also align stock returns
            self.returns = self.returns.reindex(final_dates).fillna(0)

            print(f"Successfully created factor dataset:")
            print(f"- {len(self.factors)} observations")
            print(f"- Date range: {self.factors.index[0]} to {self.factors.index[-1]}")
            print(f"- Factors: {list(self.factors.columns)}")

            return self.factors

        except Exception as e:
            print(f"Error creating final factor dataframe: {e}")
            return None

    def fetch_fama_french_factors(self, start_date="2020-01-01", end_date=None):
        """
        Alternative: Fetch actual Fama-French factors from online source
        Note: This would require additional data source like WRDS or Ken French's website
        For now, we'll use our market-based proxies
        """
        print("Using market-based factor proxies. For official Fama-French factors,")
        print("consider subscribing to WRDS or downloading from Ken French's website.")
        return self.create_factors_from_market_data()

    def run_factor_regression(self, stock_returns, factor_subset=None):
        """
        Run factor regression for each stock using specified factors
        """
        if factor_subset is None:
            factor_subset = ['Mkt-RF', 'SMB', 'HML']  # Classic FF3

        # Prepare factor data
        X_factors = self.factors[factor_subset].copy()

        results = {}

        for stock in stock_returns.columns:
            # Excess returns (subtract risk-free rate)
            y = stock_returns[stock] - self.factors['RF']

            # Align data
            common_idx = y.dropna().index.intersection(X_factors.dropna().index)
            if len(common_idx) < 50:  # Need minimum observations
                continue

            y_clean = y.loc[common_idx]
            X_clean = X_factors.loc[common_idx]

            # Add constant for alpha
            X = sm.add_constant(X_clean)

            try:
                # Run regression
                model = sm.OLS(y_clean, X).fit()

                # Calculate information ratio and other metrics
                alpha = model.params['const']
                alpha_tstat = model.tvalues['const']
                r_squared = model.rsquared
                adj_r_squared = model.rsquared_adj

                # Factor loadings
                factor_betas = model.params[factor_subset]
                factor_tstats = model.tvalues[factor_subset]

                # Tracking error (residual volatility)
                tracking_error = np.sqrt(np.var(model.resid) * 252)

                # Information ratio
                info_ratio = alpha * np.sqrt(252) / tracking_error if tracking_error > 0 else 0

                results[stock] = {
                    'alpha': alpha,
                    'alpha_annualized': alpha * 252,
                    'alpha_tstat': alpha_tstat,
                    'alpha_pvalue': model.pvalues['const'],
                    'factor_betas': factor_betas,
                    'factor_tstats': factor_tstats,
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared,
                    'tracking_error': tracking_error,
                    'info_ratio': info_ratio,
                    'model': model,
                    'observations': len(common_idx)
                }
            except Exception as e:
                print(f"Error processing {stock}: {e}")
                continue

        return results

    def compare_factor_models(self, stock_returns=None):
        """
        Compare different factor model specifications
        """
        if stock_returns is None:
            stock_returns = self.returns

        model_specs = {
            'CAPM': ['Mkt-RF'],
            'FF3': ['Mkt-RF', 'SMB', 'HML'],
            'FF4': ['Mkt-RF', 'SMB', 'HML', 'MOM'],
            'FF5': ['Mkt-RF', 'SMB', 'HML', 'QMJ', 'LV'],
            'FF6': ['Mkt-RF', 'SMB', 'HML', 'MOM', 'QMJ', 'LV']
        }

        comparison_results = {}

        for model_name, factors in model_specs.items():
            print(f"\nRunning {model_name} model...")
            results = self.run_factor_regression(stock_returns, factors)

            if not results:
                print(f"No valid results for {model_name}")
                continue

            # Aggregate statistics
            avg_r2 = np.mean([r['r_squared'] for r in results.values()])
            avg_adj_r2 = np.mean([r['adj_r_squared'] for r in results.values()])
            avg_tracking_error = np.mean([r['tracking_error'] for r in results.values()])

            # Count significant alphas
            sig_alphas = sum(1 for r in results.values() if abs(r['alpha_tstat']) > 2)

            # Average annualized alpha
            avg_alpha = np.mean([r['alpha_annualized'] for r in results.values()])

            comparison_results[model_name] = {
                'results': results,
                'avg_r_squared': avg_r2,
                'avg_adj_r_squared': avg_adj_r2,
                'avg_tracking_error': avg_tracking_error,
                'significant_alphas': sig_alphas,
                'avg_alpha': avg_alpha,
                'factors': factors,
                'num_stocks': len(results)
            }

            print(f"  - Analyzed {len(results)} stocks")
            print(f"  - Avg R²: {avg_r2:.3f}")
            print(f"  - Significant alphas: {sig_alphas}")

        self.results = comparison_results
        return comparison_results

    def plot_model_comparison(self):
        """
        Create visualizations comparing different factor models
        """
        if not self.results:
            print("Run compare_factor_models first!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Real Market Data: Factor Model Comparison', fontsize=16, fontweight='bold')

        models = list(self.results.keys())

        # R-squared comparison
        r2_values = [self.results[m]['avg_r_squared'] for m in models]
        adj_r2_values = [self.results[m]['avg_adj_r_squared'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        axes[0, 0].bar(x - width / 2, r2_values, width, label='R²', alpha=0.8)
        axes[0, 0].bar(x + width / 2, adj_r2_values, width, label='Adj. R²', alpha=0.8)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('R-squared')
        axes[0, 0].set_title('Model Explanatory Power')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Average alpha comparison
        alpha_values = [self.results[m]['avg_alpha'] * 100 for m in models]  # Convert to %
        axes[0, 1].bar(models, alpha_values, alpha=0.8, color='coral')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Average Alpha (%)')
        axes[0, 1].set_title('Average Annualized Alpha')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Significant alphas
        sig_alpha_values = [self.results[m]['significant_alphas'] for m in models]
        axes[1, 0].bar(models, sig_alpha_values, alpha=0.8, color='lightgreen')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Number of Significant Alphas')
        axes[1, 0].set_title('Significant Alphas (|t-stat| > 2)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Factor loadings heatmap for FF3 model
        if 'FF3' in self.results and self.results['FF3']['results']:
            ff3_results = self.results['FF3']['results']
            factor_betas = pd.DataFrame({
                stock: result['factor_betas']
                for stock, result in ff3_results.items()
            }).T

            im = axes[1, 1].imshow(factor_betas.values, cmap='RdYlBu', aspect='auto')
            axes[1, 1].set_xticks(range(len(factor_betas.columns)))
            axes[1, 1].set_xticklabels(factor_betas.columns)
            axes[1, 1].set_yticks(range(0, len(factor_betas), max(1, len(factor_betas) // 10)))
            axes[1, 1].set_yticklabels(
                [factor_betas.index[i] for i in range(0, len(factor_betas), max(1, len(factor_betas) // 10))])
            axes[1, 1].set_title('FF3 Factor Loadings Heatmap')
            plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

    def analyze_factor_performance(self):
        """
        Analyze individual factor performance and correlations using real data
        """
        if self.factors is None:
            print("Create factors first!")
            return

        # Factor statistics
        factor_stats = self.factors.describe()

        # Annualized statistics
        factor_annual = self.factors * 252  # Annualize
        factor_annual_stats = pd.DataFrame({
            'Mean': factor_annual.mean(),
            'Volatility': factor_annual.std() * np.sqrt(252 / 252),  # Already annualized
            'Sharpe': factor_annual.mean() / (factor_annual.std() * np.sqrt(252 / 252))
        })

        # Factor correlations
        factor_corr = self.factors.corr()

        # Plot factor analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Real Market Factor Analysis', fontsize=16, fontweight='bold')

        # Cumulative returns
        factor_cumret = (1 + self.factors[['Mkt-RF', 'SMB', 'HML', 'MOM']]).cumprod()
        for col in factor_cumret.columns:
            axes[0, 0].plot(factor_cumret.index, factor_cumret[col], label=col, linewidth=2)
        axes[0, 0].set_title('Cumulative Factor Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylabel('Cumulative Return')

        # Factor Sharpe ratios
        sharpe_ratios = factor_annual_stats['Sharpe'][['Mkt-RF', 'SMB', 'HML', 'MOM']]
        axes[0, 1].bar(sharpe_ratios.index, sharpe_ratios.values, alpha=0.8)
        axes[0, 1].set_title('Factor Sharpe Ratios')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Factor correlation heatmap
        im = axes[1, 0].imshow(factor_corr.values, cmap='RdYlBu', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(factor_corr.columns)))
        axes[1, 0].set_xticklabels(factor_corr.columns, rotation=45)
        axes[1, 0].set_yticks(range(len(factor_corr.columns)))
        axes[1, 0].set_yticklabels(factor_corr.columns)
        axes[1, 0].set_title('Factor Correlation Matrix')

        # Add correlation values to heatmap
        for i in range(len(factor_corr.columns)):
            for j in range(len(factor_corr.columns)):
                axes[1, 0].text(j, i, f'{factor_corr.iloc[i, j]:.2f}',
                                ha='center', va='center',
                                color='white' if abs(factor_corr.iloc[i, j]) > 0.5 else 'black')

        plt.colorbar(im, ax=axes[1, 0])

        # Rolling Sharpe ratio of market factor
        rolling_sharpe = (self.factors['Mkt-RF'].rolling(60).mean() * 252) / (
                    self.factors['Mkt-RF'].rolling(60).std() * np.sqrt(252))
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        axes[1, 1].set_title('60-Day Rolling Sharpe: Market Factor')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        print("\nFactor Annual Statistics:")
        print(factor_annual_stats.round(3))

        return factor_stats, factor_corr

    def generate_factor_report(self, model_name='FF3'):
        """
        Generate a comprehensive report for a specific model using real data
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found. Run compare_factor_models first!")
            return

        model_results = self.results[model_name]

        print(f"\n{'=' * 70}")
        print(f"REAL MARKET DATA: FACTOR MODEL ANALYSIS - {model_name}")
        print(f"{'=' * 70}")

        print(f"\nFactors included: {', '.join(model_results['factors'])}")
        print(f"Number of stocks analyzed: {model_results['num_stocks']}")
        print(
            f"Analysis period: {self.factors.index[0].strftime('%Y-%m-%d')} to {self.factors.index[-1].strftime('%Y-%m-%d')}")

        print(f"\nMODEL PERFORMANCE SUMMARY:")
        print(f"Average R-squared: {model_results['avg_r_squared']:.4f}")
        print(f"Average Adjusted R-squared: {model_results['avg_adj_r_squared']:.4f}")
        print(f"Average Tracking Error: {model_results['avg_tracking_error']:.4f}")
        print(f"Average Annualized Alpha: {model_results['avg_alpha']:.2%}")
        print(f"Stocks with Significant Alpha: {model_results['significant_alphas']}")

        # Detailed stock analysis
        results = model_results['results']

        if not results:
            print("No valid results to display")
            return

        # Best and worst alpha stocks
        alpha_sorted = sorted(results.items(), key=lambda x: x[1]['alpha_annualized'], reverse=True)

        print(f"\nTOP 5 ALPHA GENERATORS:")
        for i, (stock, result) in enumerate(alpha_sorted[:5]):
            print(f"{i + 1}. {stock}: Alpha = {result['alpha_annualized']:.2%} "
                  f"(t-stat: {result['alpha_tstat']:.2f}, R² = {result['r_squared']:.3f})")

        print(f"\nWORST 5 ALPHA GENERATORS:")
        for i, (stock, result) in enumerate(alpha_sorted[-5:]):
            print(f"{i + 1}. {stock}: Alpha = {result['alpha_annualized']:.2%} "
                  f"(t-stat: {result['alpha_tstat']:.2f}, R² = {result['r_squared']:.3f})")

        # Factor loading analysis
        all_betas = pd.DataFrame({
            stock: result['factor_betas']
            for stock, result in results.items()
        }).T

        print(f"\nFACTOR LOADING STATISTICS:")
        print(all_betas.describe().round(4))

        # Top factor exposures
        print(f"\nHIGHEST FACTOR EXPOSURES:")
        for factor in model_results['factors']:
            top_stock = all_betas[factor].abs().idxmax()
            exposure = all_betas.loc[top_stock, factor]
            print(f"{factor}: {top_stock} ({exposure:.3f})")

        return model_results


# Example usage with real market data
if __name__ == "__main__":
    print("Real Market Data: Quantitative Factor Model Analysis")
    print("=" * 60)

    # Initialize the model
    factor_model = QuantitativeFactorModel()

    # Fetch real market data (you can customize the stock list and date range)
    print("\n1. FETCHING REAL MARKET DATA")
    stock_returns = factor_model.fetch_market_data(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'XOM', 'WMT',
                 'NVDA', 'META', 'UNH', 'HD', 'MA', 'PFE', 'BAC', 'ABBV', 'DIS', 'NFLX'],
        start_date="2020-01-01"
    )

    # Create factors from market data
    print("\n2. CREATING FACTOR PROXIES")
    factors = factor_model.create_factors_from_market_data()

    # Analyze factors
    print("\n3. FACTOR PERFORMANCE ANALYSIS")
    factor_stats, factor_corr = factor_model.analyze_factor_performance()

    # Compare different models
    print("\n4. COMPARING FACTOR MODELS")
    comparison_results = factor_model.compare_factor_models()

    # Visualize results
    print("\n5. GENERATING VISUALIZATIONS")
    factor_model.plot_model_comparison()

    # Generate detailed reports
    print("\n6. DETAILED ANALYSIS REPORTS")
    for model in ['CAPM', 'FF3', 'FF4']:
        if model in comparison_results:
            factor_model.generate_factor_report(model)

    print("\n" + "=" * 60)
    print("REAL MARKET ANALYSIS COMPLETE")
    print("=" * 60)