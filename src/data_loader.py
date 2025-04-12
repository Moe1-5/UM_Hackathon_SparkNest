import pandas as pd
from abc import ABC, abstractmethod
import os
import logging
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AbstractDataLoader(ABC):

    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def _validate_and_standardize(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
             raise ValueError("DataFrame index must be a DatetimeIndex.")

        df.columns = df.columns.str.lower()
        required_columns_lower = [col.lower() for col in required_columns]

        missing_cols = [col for col in required_columns_lower if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")

        for col in required_columns_lower:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except ValueError as e:
                    logging.error(f"Could not convert column '{col}' to numeric: {e}")

        if not df.index.is_monotonic_increasing:
            logging.warning("Data not sorted by timestamp. Sorting now.")
            df.sort_index(inplace=True)

        initial_nan_counts = df[required_columns_lower].isnull().sum()
        total_initial_nans = initial_nan_counts.sum()
        if total_initial_nans > 0:
            logging.warning(f"Initial NaN values found in required columns before ffill:\n{initial_nan_counts[initial_nan_counts > 0]}")

        df[required_columns_lower] = df[required_columns_lower].ffill()
        logging.info(f"Forward filled NaNs in columns: {required_columns_lower}")

        remaining_nan_counts = df[required_columns_lower].isnull().sum()
        total_remaining_nans = remaining_nan_counts.sum()
        if total_remaining_nans > 0:
            logging.warning(f"NaN values remaining in required columns after ffill (likely at the start):\n{remaining_nan_counts[remaining_nan_counts > 0]}")

        logging.info(f"Data validation successful. Shape: {df.shape}. Date range: {df.index.min()} to {df.index.max()}")
        return df


class CsvDataLoader(AbstractDataLoader):
    DEFAULT_REQUIRED_COLUMNS = ['GN_BTC_ClosePrice']

    def __init__(self,
                 file_path: str,
                 timestamp_col: str = 'start_datetime',
                 datetime_format: Optional[str] = '%m/%d/%Y %H:%M',
                 required_columns: Optional[List[str]] = None,
                 csv_read_options: Optional[Dict[str, Any]] = None):
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"CSV file not found at path: {file_path}")
        self.file_path = file_path
        self.timestamp_col = timestamp_col
        self.datetime_format = datetime_format
        self.required_columns = required_columns if required_columns is not None else self.DEFAULT_REQUIRED_COLUMNS
        self.csv_read_options = csv_read_options or {}

        logging.info(f"Initialized CsvDataLoader for file: {self.file_path}")
        logging.info(f"Timestamp column: '{self.timestamp_col}', Format: '{self.datetime_format}'")
        logging.info(f"Required columns: {self.required_columns}")


    def load_data(self) -> pd.DataFrame:
        logging.info(f"Loading data from {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path, **self.csv_read_options)

            if self.timestamp_col not in df.columns:
                raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in CSV. Available: {list(df.columns)}")

            try:
                df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], format=self.datetime_format, errors='coerce')
            except Exception as e:
                 logging.error(f"Failed to parse timestamp column '{self.timestamp_col}' with format '{self.datetime_format}': {e}")
                 raise ValueError(f"Timestamp parsing failed for column '{self.timestamp_col}'.") from e

            initial_rows = len(df)
            df.dropna(subset=[self.timestamp_col], inplace=True)
            if len(df) < initial_rows:
                logging.warning(f"Dropped {initial_rows - len(df)} rows due to invalid timestamps.")

            if df.empty:
                raise ValueError("No valid data remaining after timestamp parsing.")

            df.set_index(self.timestamp_col, inplace=True)

            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"Index is not a DatetimeIndex after processing column '{self.timestamp_col}'.")

            logging.info(f"Successfully loaded {len(df)} rows from CSV.")

            df_standardized = self._validate_and_standardize(df, self.required_columns)

            critical_cols_check = [col.lower() for col in self.DEFAULT_REQUIRED_COLUMNS]
            if df_standardized[critical_cols_check].isnull().values.any():
                 rows_before = len(df_standardized)
                 df_standardized.dropna(subset=critical_cols_check, inplace=True)
                 rows_after = len(df_standardized)
                 if rows_after < rows_before:
                     logging.warning(f"Dropped {rows_before - rows_after} initial rows due to remaining NaNs in critical columns: {critical_cols_check}")

            if df_standardized.empty:
                 raise ValueError("DataFrame is empty after cleaning NaNs in critical columns.")

            return df_standardized

        except FileNotFoundError:
            logging.error(f"File not found: {self.file_path}")
            raise
        except ValueError as e:
            logging.error(f"Data loading/validation error: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during CSV loading: {e}")
            raise
