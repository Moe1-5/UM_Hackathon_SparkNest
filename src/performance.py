import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels library not found. Alpha and Beta metrics will not be calculated. Install with: pip install statsmodels")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PerformanceCalculator:

    def __init__(self,
                 portfolio_history: pd.DataFrame,
                 trade_log: pd.DataFrame,
                 initial_capital: float,
                 risk_free_rate: float = 0.0,
                 benchmark_data: Optional[pd.Series] = None,
                 data_frequency: str = 'D'
                ):
        if portfolio_history.empty:
            raise ValueError("Portfolio history cannot be empty.")
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")

        self.history = portfolio_history.copy()
        self.trades = trade_log.copy()
        self.initial_capital = initial_capital
        self.risk_free_rate_ann = risk_free_rate
        self.benchmark = benchmark_data.copy() if benchmark_data is not None else None
        self.frequency = data_frequency.upper()

        if self.frequency == 'D':
            self.periods_per_year = 252
        elif self.frequency == 'H':
            self.periods_per_year = 252 * 24
        elif self.frequency in ['T', 'MIN']:
             self.periods_per_year = 252 * 24 * 60
        else:
            logging.warning(f"Unsupported data frequency '{self.frequency}'. Defaulting annualization to 252 periods/year. Provide 'D', 'H', or 'T'/'MIN'.")
            self.periods_per_year = 252

        logging.info(f"Annualizing using {self.periods_per_year} periods per year (Frequency: {self.frequency})")
        self.risk_free_rate_period = (1 + self.risk_free_rate_ann)**(1/self.periods_per_year) - 1

        self.metrics: Dict[str, Any] = {}
        self.paired_trades: pd.DataFrame = pd.DataFrame()

        logging.info("PerformanceCalculator initialized.")

    def _pair_trades_fifo(self) -> pd.DataFrame:
        if self.trades.empty:
            logging.warning("Trade log is empty, cannot pair trades.")
            return pd.DataFrame()

        clean_trades = self.trades.dropna(subset=['timestamp', 'asset', 'type', 'quantity', 'price'])
        if clean_trades.empty:
             logging.warning("Trade log contains no valid trades after dropping NaNs.")
             return pd.DataFrame()

        paired_trades_list = []
        open_positions = {}

        log = clean_trades.sort_values(by='timestamp').itertuples()

        for trade in log:
            asset = trade.asset
            ts = trade.timestamp
            trade_type = trade.type.upper()
            qty = trade.quantity
            price = trade.price

            if qty <= 1e-9 or price <= 0:
                 logging.warning(f"Skipping invalid trade record during pairing: {trade}")
                 continue

            if asset not in open_positions:
                open_positions[asset] = []

            if trade_type == 'BUY':
                open_positions[asset].append({'ts': ts, 'price': price, 'qty': qty})

            elif trade_type == 'SELL':
                qty_to_close = qty
                if not open_positions.get(asset):
                     logging.warning(f"[{ts}] Attempted to SELL {qty:.4f} {asset}, but no open BUY position found (FIFO). Ignoring.")
                     continue

                while qty_to_close > 1e-9 and open_positions[asset]:
                    entry_trade = open_positions[asset][0]
                    entry_ts = entry_trade['ts']
                    entry_price = entry_trade['price']
                    entry_qty = entry_trade['qty']

                    if entry_price <= 0:
                         logging.error(f"Invalid entry price ({entry_price}) found for trade starting at {entry_ts}. Removing this broken entry.")
                         open_positions[asset].pop(0)
                         continue

                    close_this_qty = min(qty_to_close, entry_qty)

                    pnl = (price - entry_price) * close_this_qty
                    cost_basis = entry_price * close_this_qty
                    return_pct = (pnl / cost_basis) * 100 if cost_basis != 0 else 0
                    duration = ts - entry_ts

                    paired_trades_list.append({
                        'asset': asset,
                        'entry_ts': entry_ts,
                        'exit_ts': ts,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'quantity': close_this_qty,
                        'pnl': pnl,
                        'return_pct': return_pct,
                        'duration': duration
                    })

                    qty_to_close -= close_this_qty
                    entry_trade['qty'] -= close_this_qty

                    if entry_trade['qty'] < 1e-9:
                        open_positions[asset].pop(0)

        for asset, entries in open_positions.items():
            if entries:
                logging.info(f"Asset '{asset}' has {len(entries)} open trade(s) remaining at end of backtest (not included in paired stats).")


        if not paired_trades_list:
             logging.warning("No trades could be paired.")
             return pd.DataFrame()

        paired_df = pd.DataFrame(paired_trades_list)
        return paired_df

    def _calculate_drawdowns(self, equity_curve: pd.Series):
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak.replace(0, np.nan)
        drawdown = drawdown.fillna(0)
        max_drawdown = drawdown.min()

        drawdown_periods = drawdown[drawdown < -1e-9]
        avg_drawdown_pct = 0
        avg_drawdown_duration = pd.Timedelta(0)
        max_drawdown_duration = pd.Timedelta(0)

        if not drawdown_periods.empty:
            is_in_drawdown = drawdown < -1e-9
            drawdown_blocks = (~is_in_drawdown).cumsum()
            drawdown_periods_grouped = drawdown[is_in_drawdown].groupby(drawdown_blocks[is_in_drawdown])

            if not drawdown_periods_grouped.groups:
                 logging.warning("Could not group drawdown periods.")
            else:
                drawdown_info = []
                for _, period_indices in drawdown_periods_grouped.groups.items():
                    if len(period_indices) < 1: continue

                    start_index = period_indices[0]
                    peak_loc = drawdown.index.get_loc(start_index)
                    peak_index = drawdown.index[peak_loc - 1] if peak_loc > 0 else start_index

                    end_index = period_indices[-1]
                    duration = end_index - peak_index

                    peak_val = peak.loc[peak_index]
                    trough_val = equity_curve.loc[period_indices].min()

                    dd_value = (trough_val - peak_val) / peak_val if peak_val > 1e-9 else 0

                    drawdown_info.append({
                        'peak_ts': peak_index,
                        'trough_ts': equity_curve.loc[period_indices].idxmin(),
                        'end_ts': end_index,
                        'duration': duration,
                        'drawdown': dd_value
                    })


                if drawdown_info:
                    dd_df = pd.DataFrame(drawdown_info)
                    avg_drawdown_pct = dd_df['drawdown'].mean() * 100
                    avg_drawdown_duration = dd_df['duration'].mean()
                    max_drawdown_duration = dd_df['duration'].max()
                else:
                    logging.warning("Drawdown info list was empty after processing groups.")


        self.metrics['Max. Drawdown [%]'] = max_drawdown * 100
        self.metrics['Avg. Drawdown [%]'] = avg_drawdown_pct
        self.metrics['Max. Drawdown Duration'] = max_drawdown_duration
        self.metrics['Avg. Drawdown Duration'] = avg_drawdown_duration


    def _format_metrics(self) -> pd.Series:
        metrics_series = pd.Series(self.metrics)

        formatted_metrics = {}
        for key, value in metrics_series.items():
            if pd.isna(value):
                formatted_metrics[key] = "NaN"
            elif isinstance(value, (float, np.floating)):
                if "[$]" in key:
                     formatted_metrics[key] = f"{value:,.2f}"
                elif "[%]" in key:
                     formatted_metrics[key] = f"{value:.2f}%"
                elif key in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Profit Factor", "Beta", "SQN"]:
                     formatted_metrics[key] = f"{value:.2f}"
                elif key == "Alpha":
                     formatted_metrics[key] = f"{value:.2f}"
                else:
                    formatted_metrics[key] = f"{value:,.4f}"

            elif isinstance(value, pd.Timedelta):
                 formatted_metrics[key] = str(value).split('.')[0]

            elif isinstance(value, pd.Timestamp):
                 formatted_metrics[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(value, (int, np.integer)):
                 formatted_metrics[key] = f"{value:,}"
            else:
                 formatted_metrics[key] = str(value)

        return pd.Series(formatted_metrics).sort_index()

    def calculate_metrics(self) -> pd.Series:
        equity_curve = self.history['total_value']
        if equity_curve.empty:
            logging.error("Cannot calculate metrics: Equity curve is empty.")
            return pd.Series(dtype=object)

        equity_curve = pd.to_numeric(equity_curve, errors='coerce')
        equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        self.metrics['Start'] = equity_curve.index.min()
        self.metrics['End'] = equity_curve.index.max()
        self.metrics['Duration'] = self.metrics['End'] - self.metrics['Start']

        self.metrics['Initial Capital [$]'] = self.initial_capital
        self.metrics['Equity Final [$]'] = equity_curve.iloc[-1] if not equity_curve.empty else self.initial_capital
        self.metrics['Equity Peak [$]'] = equity_curve.max() if not equity_curve.empty else self.initial_capital
        total_return = (self.metrics['Equity Final [$]'] / self.initial_capital) - 1 if self.initial_capital > 0 else 0
        self.metrics['Return [%]'] = total_return * 100

        if self.benchmark is not None and not self.benchmark.empty:
             benchmark_numeric = pd.to_numeric(self.benchmark, errors='coerce')
             benchmark_numeric = benchmark_numeric.replace([np.inf, -np.inf], np.nan)
             aligned_benchmark = benchmark_numeric.reindex(equity_curve.index).ffill().bfill()
             aligned_benchmark = aligned_benchmark.dropna()

             if not aligned_benchmark.empty and aligned_benchmark.iloc[0] != 0:
                 benchmark_start = aligned_benchmark.iloc[0]
                 benchmark_end = aligned_benchmark.iloc[-1]
                 self.metrics['Buy & Hold Return [%]'] = ((benchmark_end / benchmark_start) - 1) * 100
             else:
                 self.metrics['Buy & Hold Return [%]'] = np.nan
        else:
            self.metrics['Buy & Hold Return [%]'] = np.nan

        years = max(self.metrics['Duration'].days / 365.25, 1/365.25)

        if self.initial_capital > 0 and self.metrics['Equity Final [$]'] > 0:
             self.metrics['CAGR [%]'] = ((self.metrics['Equity Final [$]'] / self.initial_capital)**(1 / years) - 1) * 100
        else:
             self.metrics['CAGR [%]'] = np.nan

        periodic_returns = equity_curve.pct_change().dropna()
        periodic_returns = periodic_returns.replace([np.inf, -np.inf], np.nan).dropna()

        if not periodic_returns.empty:
            mean_periodic_return = periodic_returns.mean()
            base = 1 + mean_periodic_return
            if base <= 0 :
                 self.metrics['Return (Ann.) [%]'] = -100.0
                 logging.warning("Mean periodic return <= -100%, annualized return set to -100%.")
            else:
                 self.metrics['Return (Ann.) [%]'] = (base**self.periods_per_year - 1) * 100

            volatility_periodic = periodic_returns.std()
            annualized_volatility = volatility_periodic * np.sqrt(self.periods_per_year)
            self.metrics['Volatility (Ann.) [%]'] = annualized_volatility * 100
        else:
            self.metrics['Return (Ann.) [%]'] = 0.0
            self.metrics['Volatility (Ann.) [%]'] = 0.0
            annualized_volatility = 0.0

        self._calculate_drawdowns(equity_curve)

        ann_return = self.metrics['Return (Ann.) [%]'] / 100
        max_dd = self.metrics['Max. Drawdown [%]'] / 100

        if annualized_volatility > 1e-9:
            self.metrics['Sharpe Ratio'] = (ann_return - self.risk_free_rate_ann) / annualized_volatility
        else:
            self.metrics['Sharpe Ratio'] = np.nan

        downside_returns = periodic_returns[periodic_returns < self.risk_free_rate_period]
        if not downside_returns.empty:
            downside_std_dev_periodic = downside_returns.std()
            if pd.notna(downside_std_dev_periodic) and downside_std_dev_periodic > 1e-9:
                annualized_downside_deviation = downside_std_dev_periodic * np.sqrt(self.periods_per_year)
                self.metrics['Sortino Ratio'] = (ann_return - self.risk_free_rate_ann) / annualized_downside_deviation
            else:
                 self.metrics['Sortino Ratio'] = np.nan
        else:
             self.metrics['Sortino Ratio'] = np.nan if ann_return <= self.risk_free_rate_ann else np.inf


        if max_dd < -1e-9:
            self.metrics['Calmar Ratio'] = ann_return / abs(max_dd)
        else:
            self.metrics['Calmar Ratio'] = np.nan

        if STATSMODELS_AVAILABLE and self.benchmark is not None and not periodic_returns.empty:
             benchmark_numeric = pd.to_numeric(self.benchmark, errors='coerce')
             benchmark_numeric = benchmark_numeric.replace([np.inf, -np.inf], np.nan)
             aligned_benchmark = benchmark_numeric.reindex(equity_curve.index).ffill().bfill()
             benchmark_returns = aligned_benchmark.pct_change().dropna()
             benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()

             df_aligned = pd.DataFrame({'strat': periodic_returns, 'bench': benchmark_returns}).dropna()

             if len(df_aligned) > 2:
                y = df_aligned['strat'] - self.risk_free_rate_period
                X = df_aligned['bench'] - self.risk_free_rate_period
                X = sm.add_constant(X)

                try:
                    model = sm.OLS(y, X, missing='drop').fit()
                    alpha_period = model.params.get('const', np.nan)
                    if pd.notna(alpha_period) and (1 + alpha_period) > 0:
                         self.metrics['Alpha'] = ((1 + alpha_period)**self.periods_per_year - 1) * 100
                    else:
                         self.metrics['Alpha'] = np.nan
                    self.metrics['Beta'] = model.params.get('bench', np.nan)
                except Exception as e:
                    logging.error(f"Failed to calculate Alpha/Beta via OLS: {e}")
                    self.metrics['Alpha'] = np.nan
                    self.metrics['Beta'] = np.nan
             else:
                logging.warning(f"Not enough aligned data points ({len(df_aligned)}) to calculate Alpha/Beta.")
                self.metrics['Alpha'] = np.nan
                self.metrics['Beta'] = np.nan
        else:
            self.metrics['Alpha'] = np.nan
            self.metrics['Beta'] = np.nan


        self.paired_trades = self._pair_trades_fifo()
        num_trades = len(self.paired_trades)
        self.metrics['# Trades'] = num_trades

        if num_trades > 0:
            self.paired_trades['pnl'] = pd.to_numeric(self.paired_trades['pnl'], errors='coerce')
            self.paired_trades['return_pct'] = pd.to_numeric(self.paired_trades['return_pct'], errors='coerce')
            valid_trades = self.paired_trades.dropna(subset=['pnl', 'return_pct'])
            num_valid_trades = len(valid_trades)

            if num_valid_trades > 0:
                 self.metrics['Win Rate [%]'] = (valid_trades['pnl'] > 1e-9).mean() * 100
                 self.metrics['Best Trade [%]'] = valid_trades['return_pct'].max()
                 self.metrics['Worst Trade [%]'] = valid_trades['return_pct'].min()
                 self.metrics['Avg. Trade [%]'] = valid_trades['return_pct'].mean()

                 self.metrics['Max. Trade Duration'] = valid_trades['duration'].max()
                 self.metrics['Avg. Trade Duration'] = valid_trades['duration'].mean()

                 winning_trades = valid_trades[valid_trades['pnl'] > 1e-9]['pnl']
                 losing_trades = valid_trades[valid_trades['pnl'] <= 1e-9]['pnl']

                 total_profit = winning_trades.sum()
                 total_loss = abs(losing_trades.sum())

                 if total_loss > 1e-9:
                     self.metrics['Profit Factor'] = total_profit / total_loss
                 else:
                     self.metrics['Profit Factor'] = np.inf if total_profit > 1e-9 else 1.0

                 expectancy_usd = valid_trades['pnl'].mean()
                 self.metrics['Expectancy [$]'] = expectancy_usd
                 self.metrics['Expectancy [%]'] = (expectancy_usd / self.initial_capital) * 100 if self.initial_capital else np.nan

                 pnl_std_dev = valid_trades['pnl'].std()
                 if pd.notna(pnl_std_dev) and pnl_std_dev > 1e-9:
                     self.metrics['SQN'] = np.sqrt(num_valid_trades) * expectancy_usd / pnl_std_dev
                 else:
                     self.metrics['SQN'] = np.nan
            else:
                 self.metrics['Win Rate [%]'] = np.nan
                 self.metrics['Profit Factor'] = np.nan
                 self.metrics['Expectancy [$]'] = np.nan
                 self.metrics['Expectancy [%]'] = np.nan
                 self.metrics['SQN'] = np.nan


            self.metrics['Kelly Criterion'] = np.nan

        else:
            self.metrics['Win Rate [%]'] = 0.0
            self.metrics['Best Trade [%]'] = 0.0
            self.metrics['Worst Trade [%]'] = 0.0
            self.metrics['Avg. Trade [%]'] = 0.0
            self.metrics['Max. Trade Duration'] = pd.Timedelta(0)
            self.metrics['Avg. Trade Duration'] = pd.Timedelta(0)
            self.metrics['Profit Factor'] = np.nan
            self.metrics['Expectancy [$]'] = 0.0
            self.metrics['Expectancy [%]'] = 0.0
            self.metrics['SQN'] = 0.0
            self.metrics['Kelly Criterion'] = np.nan


        self.metrics['Exposure Time [%]'] = np.nan

        total_data_points = len(equity_curve)
        if total_data_points > 0:
             self.metrics['Trade Frequency [%]'] = (num_trades / total_data_points) * 100
        else:
             self.metrics['Trade Frequency [%]'] = 0.0


        formatted_series = self._format_metrics()
        return formatted_series
