import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_loader import CsvDataLoader
from portfolio import Portfolio
from src.backtester import SimpleMAStrategy, MlAlphaStrategy, Backtester
from performance import PerformanceCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CSV_FILE_PATH = 'new_merged_data.csv'
TIMESTAMP_COL = 'start_datetime'
DATETIME_FORMAT = '%m/%d/%Y %H:%M'
PRICE_COLUMN_BTC = 'gn_btc_closeprice'

STRATEGY_CHOICE = 'MA'

if STRATEGY_CHOICE == 'MA':
    REQUIRED_COLUMNS = ['GN_BTC_ClosePrice']
elif STRATEGY_CHOICE == 'ML_PLACEHOLDER':
    REQUIRED_COLUMNS = [
        'GN_BTC_ClosePrice',
        'GN_BTC_SOPR',
        'inflow_total', 'outflow_total', 'reserve',
        'open_interest'
    ]
    ML_FEATURE_COLS = [col for col in REQUIRED_COLUMNS if col != 'GN_BTC_ClosePrice']
    ML_MODEL_PATH = None

ASSET_ID = 'BTCUSD'
INITIAL_CAPITAL = 100000.0
FEE_RATE = 0.0006
DATA_FREQUENCY = 'H'
RISK_FREE_RATE = 0.0

if STRATEGY_CHOICE == 'MA':
    SHORT_WINDOW = 10
    LONG_WINDOW = 30

TRADE_QUANTITY = 0.1

if __name__ == "__main__":
    logging.info("--- Starting Backtest ---")

    try:
        data_loader = CsvDataLoader(
            file_path=CSV_FILE_PATH, timestamp_col=TIMESTAMP_COL,
            datetime_format=DATETIME_FORMAT, required_columns=REQUIRED_COLUMNS
        )
        original_data = data_loader.load_data()
        logging.info(f"Data loaded successfully. Shape: {original_data.shape}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}", exc_info=True)
        exit()

    portfolio = Portfolio(initial_capital=INITIAL_CAPITAL, fee_rate=FEE_RATE)

    if STRATEGY_CHOICE == 'MA':
        strategy = SimpleMAStrategy(
            asset_id=ASSET_ID, price_column=PRICE_COLUMN_BTC,
            short_window=SHORT_WINDOW, long_window=LONG_WINDOW
        )
        logging.info("Using Simple Moving Average Strategy.")
    elif STRATEGY_CHOICE == 'ML_PLACEHOLDER':
        if 'ML_FEATURE_COLS' not in locals() or not ML_FEATURE_COLS:
             logging.error("ML_FEATURE_COLS not defined for ML_PLACEHOLDER strategy.")
             exit()
        strategy = MlAlphaStrategy(
            asset_id=ASSET_ID,
            feature_columns=ML_FEATURE_COLS,
            model_path=ML_MODEL_PATH
        )
        logging.info("Using ML Alpha Strategy (Placeholder).")
        logging.warning("Ensure the ML model is trained and MlAlphaStrategy is fully implemented!")

    else:
        logging.error(f"Invalid STRATEGY_CHOICE: {STRATEGY_CHOICE}")
        exit()


    asset_price_map = {ASSET_ID: PRICE_COLUMN_BTC}
    backtester = Backtester(
        data=original_data, strategy=strategy,
        portfolio=portfolio, asset_price_columns=asset_price_map
    )

    logging.info("Running backtest...")
    final_portfolio, predictions_df = backtester.run(
        trade_quantity_logic='fixed', fixed_trade_quantity=TRADE_QUANTITY
    )
    logging.info("Backtest finished.")

    logging.info("Calculating performance metrics...")
    history_df = final_portfolio.get_history_df()
    trades_df = final_portfolio.get_trade_log_df()

    if history_df.empty:
         logging.error("Portfolio history is empty. Cannot calculate performance.")
    else:
         benchmark_series = original_data.get(PRICE_COLUMN_BTC, None)
         if benchmark_series is None:
              logging.warning(f"Benchmark column '{PRICE_COLUMN_BTC}' not found...")

         calculator = PerformanceCalculator(
              portfolio_history=history_df, trade_log=trades_df,
              initial_capital=INITIAL_CAPITAL, risk_free_rate=RISK_FREE_RATE,
              benchmark_data=benchmark_series, data_frequency=DATA_FREQUENCY
         )
         try:
              performance_summary = calculator.calculate_metrics()
              print("\n--- Performance Summary ---")
              with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                   print(performance_summary)
              paired_trades_df = calculator.paired_trades

         except Exception as e:
              logging.error(f"Failed to calculate performance metrics: {e}", exc_info=True)
              performance_summary = None
              paired_trades_df = pd.DataFrame()


    logging.info("Generating visualizations...")
    if performance_summary is None:
         logging.error("Performance summary calculation failed. Cannot generate full plots.")
         if not history_df.empty:
              try:
                   plt.figure(figsize=(12, 6))
                   plt.plot(history_df.index, history_df['total_value'], label='Portfolio Value', color='blue')
                   plt.ylabel('Portfolio Value ($)')
                   plt.title('Portfolio Equity Curve (Performance Calc Failed)')
                   plt.grid(True)
                   plt.legend()
                   plt.show()
              except Exception as plot_err:
                   logging.error(f"Failed to plot equity curve: {plot_err}")
         exit()


    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

        if not history_df.empty:
            axes[0].plot(history_df.index, history_df['total_value'], label='Portfolio Value', color='blue')
            axes[0].set_ylabel('Portfolio Value ($)')
            axes[0].set_title('Portfolio Equity Curve')
            axes[0].grid(True)
            axes[0].legend()
        else:
             axes[0].set_title('Equity Curve (No Data)')
             axes[0].grid(True)


        price_series = original_data.get(PRICE_COLUMN_BTC, None)
        if price_series is not None:
            axes[1].plot(price_series.index, price_series, label=f'{ASSET_ID} Price ({PRICE_COLUMN_BTC})', color='black', alpha=0.9, linewidth=1.0)

            if not paired_trades_df.empty:
                 buy_entries = paired_trades_df[paired_trades_df['entry_ts'].isin(price_series.index) & (paired_trades_df['asset'] == ASSET_ID)]
                 axes[1].scatter(buy_entries['entry_ts'], buy_entries['entry_price'],
                                 label='Buy Entry', marker='^', color='lime', s=50, alpha=0.8, zorder=5)

                 sell_exits = paired_trades_df[paired_trades_df['exit_ts'].isin(price_series.index) & (paired_trades_df['asset'] == ASSET_ID)]
                 axes[1].scatter(sell_exits['exit_ts'], sell_exits['exit_price'],
                                 label='Sell Exit', marker='v', color='red', s=50, alpha=0.8, zorder=5)
                 axes[1].legend()
            else:
                 axes[1].legend([f'{ASSET_ID} Price ({PRICE_COLUMN_BTC})'])


            axes[1].set_ylabel('Price ($)')
            axes[1].set_title(f'{ASSET_ID} Price and Trades (Use plot window tools to zoom/pan)')
            axes[1].grid(True)
        else:
             axes[1].set_title('Price Chart (Price Column Not Found)')
             axes[1].grid(True)


        has_predictions = (not predictions_df.empty and
                           ASSET_ID in predictions_df.columns and
                           not predictions_df[ASSET_ID].isnull().all())

        if has_predictions:
             ax3_twin = axes[2].twinx()

             if price_series is not None:
                   axes[2].plot(price_series.index, price_series, label=f'{ASSET_ID} Price', color='black', alpha=0.6)
                   axes[2].set_ylabel('Price ($)', color='black')
                   axes[2].tick_params(axis='y', labelcolor='black')
                   axes[2].grid(True, axis='y', linestyle='--', alpha=0.6)

             prediction_series = predictions_df[ASSET_ID].reindex(price_series.index, method='ffill')
             ax3_twin.plot(prediction_series.index, prediction_series, label='Model Prediction', color='purple', alpha=0.7, linestyle='--')
             ax3_twin.set_ylabel('Prediction Value', color='purple')
             ax3_twin.tick_params(axis='y', labelcolor='purple')

             axes[2].set_title(f'{ASSET_ID} Price vs. Model Prediction')
             lines, labels = axes[2].get_legend_handles_labels()
             lines2, labels2 = ax3_twin.get_legend_handles_labels()
             if lines and lines2:
                 ax3_twin.legend(lines + lines2, labels + labels2, loc='upper left')
             elif lines:
                 axes[2].legend(loc='upper left')
             elif lines2:
                 ax3_twin.legend(loc='upper left')

        else:
             axes[2].set_title('Predictions vs. Price (No valid prediction data found for this strategy run)')
             axes[2].grid(True)
             if price_series is not None:
                  axes[2].plot(price_series.index, price_series, label=f'{ASSET_ID} Price', color='black', alpha=0.6)
                  axes[2].set_ylabel('Price ($)')
                  axes[2].legend()



        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(formatter)

        fig.autofmt_xdate()
        plt.xlabel('Date / Time')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle('Backtest Analysis', fontsize=16, y=0.99)
        plt.show()

    except ImportError:
        print("\nMatplotlib not installed. Skipping visualization.")
    except Exception as e:
        print(f"\nError generating visualizations: {e}", exc_info=True)


    logging.info("--- Backtest End ---")
