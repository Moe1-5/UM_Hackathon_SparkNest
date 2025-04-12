import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod

from portfolio import Portfolio

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn or joblib not found. pip install scikit-learn joblib. ML Strategy unavailable.")
    SKLEARN_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AbstractStrategy(ABC):
    @abstractmethod
    def generate_signals(self, timestamp: pd.Timestamp, current_data: pd.DataFrame, **kwargs) -> Tuple[Dict[str, int], Dict[str, Any]]:
        raise NotImplementedError

class SimpleMAStrategy(AbstractStrategy):
    def __init__(self,
                 asset_id: str,
                 price_column: str,
                 short_window: int = 10,
                 long_window: int = 30):
        self.asset_id = asset_id
        self.price_column = price_column.lower()
        self.short_window = short_window
        self.long_window = long_window
        if short_window >= long_window:
            raise ValueError("Short window must be less than long window")
        if not price_column:
            raise ValueError("Price column name cannot be empty")
        logging.info(f"Initialized SimpleMAStrategy for {asset_id} using data column '{self.price_column}' with windows {short_window}/{long_window}")

    def generate_signals(self, timestamp: pd.Timestamp, current_data: pd.DataFrame, **kwargs) -> Tuple[Dict[str, int], Dict[str, Any]]:
        signals = {self.asset_id: 0}
        predictions = {self.asset_id: np.nan}

        if self.price_column not in current_data.columns:
             logging.error(f"[{timestamp}] Price column '{self.price_column}' not found in data for MA strategy. Cannot generate signal.")
             return signals, predictions

        if len(current_data) < self.long_window + 1:
            return signals, predictions

        try:
            price_data = pd.to_numeric(current_data[self.price_column], errors='coerce')
            if price_data.isnull().all():
                 logging.warning(f"[{timestamp}] Price column '{self.price_column}' contains no numeric data. Cannot calculate MAs.")
                 return signals, predictions

            short_ma = price_data.rolling(window=self.short_window).mean()
            long_ma = price_data.rolling(window=self.long_window).mean()
        except Exception as e:
            logging.error(f"[{timestamp}] Error calculating MAs for column '{self.price_column}': {e}")
            return signals, predictions

        try:
            latest_short_ma = short_ma.iloc[-1]
            latest_long_ma = long_ma.iloc[-1]
            prev_short_ma = short_ma.iloc[-2]
            prev_long_ma = long_ma.iloc[-2]
        except IndexError:
            logging.warning(f"[{timestamp}] Index error accessing MA values. Not enough data calculated?")
            return signals, predictions

        if pd.isna(latest_short_ma) or pd.isna(latest_long_ma) or pd.isna(prev_short_ma) or pd.isna(prev_long_ma):
            return signals, predictions

        if latest_short_ma > latest_long_ma and prev_short_ma <= prev_long_ma:
            signals[self.asset_id] = 1
            logging.debug(f"[{timestamp}] {self.asset_id} BUY signal generated (Short MA {latest_short_ma:.2f} > Long MA {latest_long_ma:.2f}) using '{self.price_column}'")

        elif latest_short_ma < latest_long_ma and prev_short_ma >= prev_long_ma:
            signals[self.asset_id] = -1
            logging.debug(f"[{timestamp}] {self.asset_id} SELL signal generated (Short MA {latest_short_ma:.2f} < Long MA {latest_long_ma:.2f}) using '{self.price_column}'")

        return signals, predictions

class MlAlphaStrategy(AbstractStrategy):
    def __init__(self,
                 asset_id: str,
                 feature_columns: List[str],
                 model_path: Optional[str] = None,
                 buy_threshold: float = 0.55,
                 sell_threshold: float = 0.45
                ):
        if not SKLEARN_AVAILABLE:
             logging.error("MlAlphaStrategy requires scikit-learn and joblib. Strategy unavailable.")

        self.asset_id = asset_id
        self.feature_columns = [col.lower() for col in feature_columns]
        self.model_path = model_path
        self.model = None
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        if self.model_path and SKLEARN_AVAILABLE:
             try:
                 self.model = joblib.load(self.model_path)
                 logging.info(f"ML model loaded successfully from {self.model_path}")
             except FileNotFoundError:
                 logging.error(f"Model file not found at path: {self.model_path}. Using placeholder logic.")
                 self.model = self._create_placeholder_model()
             except Exception as e:
                 logging.error(f"Failed to load ML model from {self.model_path}: {e}. Using placeholder logic.")
                 self.model = self._create_placeholder_model()
        elif SKLEARN_AVAILABLE:
             logging.warning("No model_path provided for MlAlphaStrategy. Using placeholder logic.")
             self.model = self._create_placeholder_model()
        else:
             logging.error("Cannot initialize MlAlphaStrategy model as scikit-learn/joblib are missing.")

        logging.info(f"Initialized MlAlphaStrategy for {asset_id} using features: {self.feature_columns}")
        if self.model:
             logging.info(f"Signal thresholds: Buy > {self.buy_threshold}, Sell < {self.sell_threshold}")

    def _create_placeholder_model(self):
         logging.warning("Creating a PLACEHOLDER ML model - generates RANDOM signals.")
         class DummyModel:
             def predict_proba(self, X):
                  n_samples = len(X)
                  proba = np.random.rand(n_samples, 2)
                  return proba / proba.sum(axis=1, keepdims=True)
             def predict(self, X):
                 proba = self.predict_proba(X)
                 return (proba[:, 1] > 0.5).astype(int)
         return DummyModel()

    def _preprocess_features(self, current_data: pd.DataFrame) -> Optional[pd.DataFrame]:
         missing_cols = [col for col in self.feature_columns if col not in current_data.columns]
         if missing_cols:
             logging.warning(f"Missing feature columns for ML model: {missing_cols}. Cannot generate features.")
             return None

         latest_features_raw = current_data[self.feature_columns].iloc[-1:]

         features_filled = latest_features_raw.ffill().fillna(0)
         if features_filled.isnull().values.any():
              logging.warning(f"NaN values remain in features after fillna(0). Skipping prediction.")
              return None

         features_processed = features_filled

         return features_processed

    def generate_signals(self, timestamp: pd.Timestamp, current_data: pd.DataFrame, **kwargs) -> Tuple[Dict[str, int], Dict[str, Any]]:
        signals = {self.asset_id: 0}
        predictions = {self.asset_id: np.nan}

        if not self.model:
             logging.error("ML model is not available (missing libraries or failed load?). Cannot generate ML signals.")
             return signals, predictions

        features = self._preprocess_features(current_data)
        if features is None or features.empty:
             logging.debug(f"[{timestamp}] Features not available or invalid for ML model. Holding.")
             return signals, predictions

        try:
            if hasattr(self.model, "predict_proba"):
                model_prediction_proba = self.model.predict_proba(features)[0]
                prob_up = model_prediction_proba[1]
                prediction_value = prob_up

                if prob_up > self.buy_threshold:
                    signals[self.asset_id] = 1
                elif prob_up < self.sell_threshold:
                     signals[self.asset_id] = -1

                logging.debug(f"[{timestamp}] {self.asset_id} ML Signal: {signals[self.asset_id]}, Prediction(ProbUp): {prediction_value:.4f}")
                predictions[self.asset_id] = prediction_value
            else:
                 logging.warning("Loaded model does not have 'predict_proba' method needed for this logic.")

        except Exception as e:
             logging.error(f"[{timestamp}] Error during ML prediction/signal generation: {e}", exc_info=False)
             signals[self.asset_id] = 0
             predictions[self.asset_id] = np.nan

        return signals, predictions

class Backtester:
    def __init__(self,
                 data: pd.DataFrame,
                 strategy: AbstractStrategy,
                 portfolio: Portfolio,
                 asset_price_columns: Dict[str, str],
                 execution_price_type: str = 'close'
                 ):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data must be sorted chronologically by timestamp.")
        if not asset_price_columns:
             raise ValueError("Asset price columns mapping cannot be empty.")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in asset_price_columns.items()):
             raise ValueError("asset_price_columns must map string asset IDs to string column names.")

        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.asset_price_columns = {k: v.lower() for k, v in asset_price_columns.items()}
        self.assets = list(self.asset_price_columns.keys())
        self.predictions_log = []

        for asset, price_col in self.asset_price_columns.items():
             if price_col not in self.data.columns:
                 raise ValueError(f"Execution/valuation price column '{price_col}' for asset '{asset}' not found in data columns: {list(self.data.columns)}")

        logging.info("Backtester initialized.")
        logging.warning("Current implementation uses closing price of the signal bar for execution. Consider adjusting for lookahead bias (e.g., execute on next bar open).")

    def run(self, trade_quantity_logic: str = 'fixed', fixed_trade_quantity: float = 1.0, **kwargs) -> Tuple[Portfolio, pd.DataFrame]:
        logging.info(f"Starting backtest run from {self.data.index.min()} to {self.data.index.max()}...")
        logging.info(f"Trading assets: {self.assets}. Trade logic: {trade_quantity_logic}, Qty: {fixed_trade_quantity}")
        self.predictions_log = []

        for timestamp, current_bar_data in self.data.iterrows():
            current_prices_for_valuation = {}
            for asset, price_col in self.asset_price_columns.items():
                 price = current_bar_data.get(price_col, np.nan)
                 if not pd.isna(price):
                     current_prices_for_valuation[asset] = price

            if current_prices_for_valuation:
                 self.portfolio.update_market_value(timestamp, pd.Series(current_prices_for_valuation))
            else:
                 self.portfolio.history.append((timestamp, self.portfolio.current_total_value))

            historical_data_view = self.data.loc[:timestamp]
            try:
                 signals, predictions = self.strategy.generate_signals(timestamp, historical_data_view)
            except Exception as e:
                 logging.error(f"[{timestamp}] Error during strategy signal generation: {e}", exc_info=True)
                 signals = {asset: 0 for asset in self.assets}
                 predictions = {asset: np.nan for asset in self.assets}

            if predictions:
                 pred_record = {'timestamp': timestamp, **predictions}
                 self.predictions_log.append(pred_record)

            for asset_id, signal in signals.items():
                if asset_id not in self.assets: continue

                if signal != 0:
                    execution_price_col_name = self.asset_price_columns.get(asset_id)
                    if not execution_price_col_name: continue

                    execution_price = current_bar_data.get(execution_price_col_name, np.nan)

                    if pd.isna(execution_price) or execution_price <= 0:
                        logging.warning(f"[{timestamp}] Cannot execute trade for {asset_id}: Invalid execution price ({execution_price}) in '{execution_price_col_name}'.")
                        continue

                    order_type = 'BUY' if signal == 1 else 'SELL'
                    qty_to_trade = 0.0

                    if trade_quantity_logic == 'fixed':
                        qty_to_trade = fixed_trade_quantity

                    if qty_to_trade <= 1e-9: continue

                    if order_type == 'SELL':
                        current_pos = self.portfolio.current_positions.get(asset_id, 0.0)
                        if current_pos < 1e-9: continue
                        qty_to_trade = min(qty_to_trade, current_pos)

                    if qty_to_trade > 1e-9:
                        self.portfolio.transact_asset(
                            timestamp=timestamp, asset=asset_id,
                            order_type=order_type, quantity=qty_to_trade,
                            price=execution_price
                        )

        logging.info("Backtest run finished.")
        final_value = self.portfolio.current_total_value
        logging.info(f"Initial Capital: ${self.portfolio.initial_capital:,.2f}")
        logging.info(f"Final Portfolio Value: ${final_value:,.2f}")

        predictions_df = pd.DataFrame(self.predictions_log)
        if not predictions_df.empty:
             try:
                 predictions_df.set_index('timestamp', inplace=True)
             except KeyError:
                  logging.error("Could not set 'timestamp' as index for predictions DataFrame.")

        return self.portfolio, predictions_df
