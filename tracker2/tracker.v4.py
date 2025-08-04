
import logging
from alpha_vantage_client import AlphaVantageClient
from config_manager import load_config
import pandas as pd
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_and_process_data(client: AlphaVantageClient, stock_list: List[str]) -> pd.DataFrame:
    """Fetch stock data and process it into a DataFrame."""
    all_data = []
    for stock in stock_list:
        logging.info(f"Fetching data for stock: {stock}")
        data = client.get_stock_data(stock)
        if data:
            logging.info(f"Data for {stock} retrieved successfully.")
            # Example processing: Convert to DataFrame
            df = pd.DataFrame(data)
            df['symbol'] = stock
            all_data.append(df)
        else:
            logging.warning(f"Failed to retrieve data for {stock}.")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        logging.error("No data retrieved for any stocks.")
        return pd.DataFrame()

def save_data_to_csv(df: pd.DataFrame, filename: str):
    """Save DataFrame to a CSV file."""
    try:
        df.to_csv(filename, index=False)
        logging.info(f"Data saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save data to {filename}: {e}")

def main():
    # Load configuration
    config = load_config('config.json')
    if not config:
        logging.error("Failed to load configuration.")
        return

    # Initialize Alpha Vantage client
    client = AlphaVantageClient(config.api_key)

    # Fetch and process data
    stock_data = fetch_and_process_data(client, config.stock_list)

    # Save processed data to CSV
    if not stock_data.empty:
        save_data_to_csv(stock_data, '/home/user/output/stock_data.csv')

if __name__ == "__main__":
    main()
