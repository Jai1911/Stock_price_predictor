import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

tickers = ['AAPL', 'NVDA', 'AMZN', 'MSFT', 'GOOGL']
end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)

print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

raw_data = yf.download(tickers, start=start_date, end=end_date)

raw_data.columns.names = ['Price', 'Ticker']
tidy_df = raw_data.stack(level='Ticker', future_stack=True).reset_index()
tidy_df = tidy_df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
tidy_df['Date'] = pd.to_datetime(tidy_df['Date']).dt.tz_localize(None)

print("Calculating features...")

tidy_df['Daily_Return'] = tidy_df.groupby('Ticker')['Close'].pct_change()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

tidy_df['RSI_14'] = tidy_df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))

tidy_df['Vol_14D'] = tidy_df.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(window=14).std())

ema12 = tidy_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26 = tidy_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
tidy_df['MACD'] = ema12 - ema26

direction = np.sign(tidy_df.groupby('Ticker')['Close'].diff()).fillna(0)
tidy_df['OBV'] = (direction * tidy_df['Volume']).groupby(tidy_df['Ticker']).cumsum()

sma_50 = tidy_df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=50).mean())
tidy_df['Dist_From_50SMA'] = (tidy_df['Close'] - sma_50) / sma_50

tidy_df['Target_Next_Day_Return'] = tidy_df.groupby('Ticker')['Daily_Return'].shift(-1)

tidy_df.dropna(inplace=True)

final_columns = [
    'Date', 'Ticker', 'Close', 'High', 'Low', 
    'Open', 'Volume', 'Daily_Return', 
    'Vol_14D', 'RSI_14', 'MACD', 'OBV', 'Dist_From_50SMA', 
    'Target_Next_Day_Return'
]
final_df = tidy_df[final_columns]

print(final_df.head(10))
print(f"\nTotal rows processed: {len(final_df)}")

output_file = 'stocks_5yr_baseline.parquet'
final_df.to_parquet(output_file, engine='pyarrow')
print(f"\nSuccessfully saved quantitative baseline to {output_file}")