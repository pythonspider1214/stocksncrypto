
import requests
import time

class AlphaVantageClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.last_request_time = 0
        self.rate_limit = 60  # 1 request per minute

    def get_stock_data(self, symbol, function="TIME_SERIES_DAILY"):
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit:
            time.sleep(self.rate_limit - (current_time - self.last_request_time))

        try:
            response = requests.get(self.base_url, params={
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key
            })
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"An error occurred: {req_err}")
        return None
