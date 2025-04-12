# Combined Data Fetching and Merging Script

import os
import pandas as pd
import asyncio
from dotenv import load_dotenv
import cybotrade_datasource
from datetime import datetime, timezone, timedelta
# from IPython.display import display # Replaced with print for wider compatibility
import time
from functools import reduce

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("API_KEY")

# --- Define Topics to Fetch ---
# Dictionary: Key = Desired Excel Sheet Name, Value = Topic String
# Reordered with potentially slower/larger datasets at the end
TOPICS_TO_FETCH = {
    # Faster endpoints first (lower frequency or smaller data)
    "GN_BTC_SOPR":             "glassnode|indicators/sopr_adjusted?a=BTC&i=24h",
    "GN_BTC_SpotVolumeDaily":  "glassnode|market/spot_volume_daily_sum?a=BTC&i=24h",
    "GN_BTC_ClosePrice":       "glassnode|market/price_usd_close?a=BTC&i=1h",
    "CG_BTC_FundingRate":      "coinglass|futures/funding_rates/history?symbol=BTC&interval=8h",

    # Medium-sized datasets
    "CQ_BTC_Exchange_Inflow":  "cryptoquant|btc/exchange-flows/inflow?exchange=binance&window=hour",
    "CQ_BTC_OI_Binance":       "cryptoquant|btc/market-data/open-interest?exchange=binance&window=hour",
    "CQ_BTC_Exchange_Reserve": "cryptoquant|btc/exchange-flows/reserve?exchange=binance&window=hour",
    "GN_BTC_PriceDrawdown":    "glassnode|market/price_drawdown_relative?a=BTC&i=1h",
    "CG_BTC_OpenInterest":     "coinglass|futures/open_interest/history?symbol=BTC&interval=1h",
    "CG_BTC_LiquidationData":  "coinglass|futures/liquidation_history?symbol=BTC&interval=1h",

    # Potentially larger datasets at the end
    "GN_BTC_NetworkActivity":  "glassnode|transactions/count?a=BTC&i=1h",
    "GN_BTC_ActiveAddresses":  "glassnode|addresses/active_count?a=BTC&i=1h",
    "CQ_BTC_Miner_Outflow":    "cryptoquant|btc/miner-flows/outflow?miner=all_miner&window=hour",
    "GN_BTC_ExchangeBalance":  "glassnode|distribution/balance_exchanges?a=BTC&i=1h",
    "GN_BTC_PriceOHLC":        "glassnode|market/price_usd_ohlc?a=BTC&i=1h",

    # Orderbook data last (typically very large)
    "CG_Spot_Orderbook_BTC":   "coinglass|spot/orderbook/history?exchange=Binance&interval=1h&symbol=BTCUSDT"
}

# Set time ranges - fetch 5 years of data
YEARS_TO_FETCH = 5
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=365*YEARS_TO_FETCH)

# Define a shorter time range for orderbook data only (30 days)
orderbook_start_time = end_time - timedelta(days=30)

# Output file paths
intermediate_excel_file = "crypto_data_5yr_intermediate.xlsx" # Renamed to avoid confusion
final_csv_output_file = 'crypto_data_merged.csv'

# Progress tracking
total_topics = len(TOPICS_TO_FETCH)
current_topic_fetch = 0 # Renamed to avoid clash

# --- Part 1: Data Fetching Logic ---

async def fetch_data_for_topic(description, topic_str, start, end):
    """Fetches data for a single topic."""
    global current_topic_fetch
    current_topic_fetch += 1

    print(f"\nFetching data for: {description} ({current_topic_fetch}/{total_topics})...")
    print(f"Topic: {topic_str}")
    print(f"Range: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} UTC")

    if not API_KEY:
        print("API Key not found. Skipping this topic.")
        return None

    start_fetch_time = time.time()
    try:
        data = await cybotrade_datasource.query_paginated(
            api_key=API_KEY,
            topic=topic_str,
            start_time=start,
            end_time=end
        )

        fetch_time = time.time() - start_fetch_time

        if data:
            record_count = len(data)
            print(f"Data received: {record_count} records in {fetch_time:.2f} seconds.")

            # Convert raw data to a DataFrame
            df = pd.DataFrame(data)

            # Print brief summary
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Columns: {', '.join(df.columns)}")

            # Process timestamp columns based on data format
            if "start_time" in df.columns:
                try:
                    # First try milliseconds (most common)
                    df["start_datetime"] = pd.to_datetime(df["start_time"], unit="ms", errors='coerce')
                    print("Attempted conversion 'start_time' to datetime using milliseconds.")
                except:
                    df["start_datetime"] = pd.NaT # Set to NaT if first attempt fails structurally

                if df["start_datetime"].isnull().all(): # Check if conversion failed
                   try:
                       # If milliseconds fail or result in all NaT, try seconds
                       df["start_datetime"] = pd.to_datetime(df["start_time"], unit="s", errors='coerce')
                       print("Attempted conversion 'start_time' to datetime using seconds.")
                   except Exception as e:
                       print(f"Error converting start_time with seconds: {e}")
                       df["start_datetime"] = pd.NaT

            # Handle timestamp "t" column if present and start_datetime wasn't created
            if "t" in df.columns and "start_datetime" not in df.columns:
                try:
                    df["start_datetime"] = pd.to_datetime(df["t"], unit="ms", errors='coerce') # Prefer 'start_datetime' name
                    print("Converted 't' column to 'start_datetime' using milliseconds.")
                except Exception as e:
                    print(f"Error converting 't' column: {e}")
                    df["start_datetime"] = pd.NaT

            # Drop rows where timestamp conversion failed
            initial_rows = len(df)
            df.dropna(subset=['start_datetime'], inplace=True)
            if len(df) < initial_rows:
                print(f"Dropped {initial_rows - len(df)} rows due to invalid timestamps.")

            # Handle Glassnode's 'v' column by giving it a better name if it exists alone
            # Check original columns *before* potential renaming or adding start_datetime
            original_cols = set(data[0].keys()) if data else set()
            if original_cols == {"start_time", "v"} or original_cols == {"t", "v"}:
                if 'v' in df.columns: # Ensure 'v' still exists after processing
                    metric_name = description.split('_')[2] if len(description.split('_')) > 2 else "value"
                    df = df.rename(columns={"v": f"{description}_value"}) # Use full description for uniqueness
                    print(f"Renamed generic 'v' column to '{description}_value'")

            # Display just the first 3 rows to save memory
            print("\nSample data (first 3 rows):")
            # Use to_string() for better display in terminal/log files
            print(df.head(3).to_string())

            return df
        else:
            print("No data received for this topic.")
            return None

    except Exception as e:
        print(f"An error occurred while fetching '{description}': {e}")
        return None
    finally:
        # Small delay between requests
        await asyncio.sleep(1.0)

async def run_data_fetching():
    """Main async function to coordinate data fetching."""
    print(f"\n--- Starting Part 1: Data Fetching ---")
    print(f"Fetching data for {len(TOPICS_TO_FETCH)} topics...")
    print(f"Time range (default): {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    print(f"Time range (orderbook): {orderbook_start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    print(f"Intermediate data will be saved to: {intermediate_excel_file}")

    # Create Excel writer - first check if file exists
    file_exists = os.path.isfile(intermediate_excel_file)

    # Set proper parameters based on whether file exists
    if file_exists:
        print(f"Intermediate file {intermediate_excel_file} already exists. Will overwrite/replace sheets.")
        writer_params = {
            'path': intermediate_excel_file,
            'engine': 'openpyxl',
            'mode': 'a', # Append mode
            'if_sheet_exists': 'replace' # Replace sheet if it exists
        }
    else:
        print(f"Creating new intermediate file: {intermediate_excel_file}")
        writer_params = {
            'path': intermediate_excel_file,
            'engine': 'openpyxl',
            'mode': 'w' # Write mode
        }

    try:
        # Create writer with appropriate parameters
        with pd.ExcelWriter(**writer_params) as writer:
            # Fetch and save each topic one at a time
            for i, (sheet_name, topic_string) in enumerate(TOPICS_TO_FETCH.items(), 1):
                print(f"\nFetching Process {i}/{total_topics}: {sheet_name}")

                # Use shorter time range only for orderbook data
                current_start_time = orderbook_start_time if "Orderbook" in sheet_name else start_time

                # Fetch data
                df = await fetch_data_for_topic(sheet_name, topic_string, current_start_time, end_time)

                if df is not None and not df.empty:
                    # Save directly to Excel immediately after fetching
                    print(f"Saving {sheet_name} to Excel...")
                    # Ensure sheet name fits Excel's 31 character limit
                    safe_sheet_name = sheet_name[:31]
                    if len(safe_sheet_name) < len(sheet_name):
                        print(f"  Warning: Sheet name '{sheet_name}' truncated to '{safe_sheet_name}' for Excel.")

                    # Drop original time columns if they exist before saving
                    df_to_save = df.copy()
                    if 'start_time' in df_to_save.columns:
                        df_to_save = df_to_save.drop('start_time', axis=1)
                    if 't' in df_to_save.columns:
                         df_to_save = df_to_save.drop('t', axis=1)

                    df_to_save.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                    print(f"Successfully saved {len(df)} rows for {sheet_name} to sheet '{safe_sheet_name}'")

                    del df # Free up memory
                    del df_to_save
                else:
                    print(f"Skipping sheet {sheet_name} - no data received or processed.")

                print(f"Intermediate progress saved to {intermediate_excel_file}")

        print("\n--- Data Fetching Part Complete ---")
        return True # Indicate success

    except Exception as e:
        print(f"\nCRITICAL ERROR during data fetching or saving to Excel: {e}")
        print("The script might not be able to proceed to the merging step.")
        return False # Indicate failure

# --- Part 2: Data Merging Logic ---

def run_data_merging():
    """Function to handle reading the Excel file and merging the data."""
    print(f"\n--- Starting Part 2: Data Merging ---")
    print(f"Reading intermediate data from: {intermediate_excel_file}")
    print(f"Final merged data will be saved to: {final_csv_output_file}")

    if not os.path.exists(intermediate_excel_file):
        print(f"Error: Intermediate file '{intermediate_excel_file}' not found. Cannot merge.")
        return False

    try:
        # Load the Excel file and get all sheet names
        xl = pd.ExcelFile(intermediate_excel_file)
        sheet_names = xl.sheet_names
        print(f"Found {len(sheet_names)} sheets in intermediate file: {', '.join(sheet_names)}")

        # Initialize an empty list to store dataframes
        dfs = []
        all_value_column_names = set() # Track only non-datetime columns for conflict

        # First pass: Process each sheet and collect value column names
        print("\nProcessing sheets from Excel file...")
        for sheet_name in sheet_names:
            print(f" Reading sheet: {sheet_name}")

            # Read the sheet into a dataframe
            try:
                 df = xl.parse(sheet_name)
            except Exception as e:
                 print(f"  Warning: Could not parse sheet '{sheet_name}'. Error: {e}. Skipping.")
                 continue

            # Check if start_datetime column exists
            if 'start_datetime' not in df.columns:
                print(f"  Warning: 'start_datetime' column not found in sheet '{sheet_name}'. Skipping this sheet.")
                continue

            # Convert start_datetime just in case it wasn't saved correctly (e.g., read as object)
            df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
            df.dropna(subset=['start_datetime'], inplace=True) # Drop rows if conversion failed here

            if df.empty:
                 print(f"  Warning: Sheet '{sheet_name}' is empty after timestamp validation. Skipping.")
                 continue

            # Identify value columns (all columns except 'start_datetime')
            value_columns = [col for col in df.columns if col != 'start_datetime']

            if not value_columns:
                 print(f"  Warning: No value columns found in sheet '{sheet_name}' besides 'start_datetime'. Skipping.")
                 continue

            # Rename columns to ensure uniqueness BEFORE merge attempt (more robust)
            # Prefix all value columns with the sheet name
            rename_dict = {col: f"{sheet_name}_{col}" for col in value_columns}
            df = df.rename(columns=rename_dict)
            print(f"  Renamed value columns with prefix '{sheet_name}_'. Example: '{list(rename_dict.values())[0]}'")

            # Keep only start_datetime and the newly renamed value columns
            df = df[['start_datetime'] + list(rename_dict.values())]

            # Add to our list of dataframes
            dfs.append(df)
            print(f"  Sheet '{sheet_name}' processed for merge. Shape: {df.shape}")

        if not dfs:
            print("\nError: No valid sheets could be processed from the Excel file. Cannot merge.")
            return False

        # Merge all dataframes using reduce for efficiency
        # Since columns are pre-emptively renamed, standard outer merge is fine
        print("\nMerging all processed sheets based on 'start_datetime'...")

        def merge_outer(left, right):
             # Simple outer merge on the timestamp column
             return pd.merge(left, right, on='start_datetime', how='outer')

        merged_df = reduce(merge_outer, dfs)
        print(f"Merging complete. Initial merged shape: {merged_df.shape}")

        # Sort by start_datetime to ensure proper time alignment
        print("Sorting merged data by 'start_datetime'...")
        merged_df = merged_df.sort_values('start_datetime')

        # Remove any fully duplicate rows (unlikely after renaming, but good practice)
        duplicate_count = merged_df.duplicated().sum()
        if duplicate_count > 0:
            print(f"Removing {duplicate_count} fully duplicate rows...")
            merged_df = merged_df.drop_duplicates()

        # Display final column names for verification
        print(f"\nFinal columns in merged DataFrame ({len(merged_df.columns)} total):")
        # Print first few and last few columns if too many
        cols_list = merged_df.columns.tolist()
        if len(cols_list) > 20:
             print(f"  {', '.join(cols_list[:10])}, ..., {', '.join(cols_list[-10:])}")
        else:
             print(f"  {', '.join(cols_list)}")


        # Save to CSV
        print(f"\nSaving final merged data to {final_csv_output_file}...")
        merged_df.to_csv(final_csv_output_file, index=False)

        print(f"\n--- Data Merging Part Complete ---")
        print(f"Final dataset shape: {merged_df.shape}")
        print(f"Data successfully saved to {final_csv_output_file}")
        return True # Indicate success

    except Exception as e:
        print(f"\nCRITICAL ERROR during data merging or saving to CSV: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return False # Indicate failure


# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Combined Data Fetching and Merging Script ---")

    if not API_KEY:
        print("\nCRITICAL ERROR: API_KEY not found in environment variables or .env file!")
        print("Please ensure your API_KEY is set correctly.")
    else:
        print(f"\nFound API_KEY (length: {len(API_KEY)}).")

        # Run Part 1: Data Fetching
        fetch_success = asyncio.run(run_data_fetching())

        # Only proceed to Part 2 if Part 1 was successful
        if fetch_success:
            # Run Part 2: Data Merging
            merge_success = run_data_merging()
            if merge_success:
                print("\n--- Script finished successfully ---")
            else:
                print("\n--- Script finished with errors during merging ---")
        else:
            print("\n--- Script finished with errors during data fetching ---")
            print("Merging step was skipped.")

    print("\n--- End of script ---")
