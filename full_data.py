import pandas as pd

print("Loading datasets...")

stocks_df = pd.read_parquet('stocks_5yr_baseline.parquet')

sentiment_df = pd.read_parquet('sentiment_1yr_finbert.parquet')

print("Executing the Master Merge...")
master_df = pd.merge(
    stocks_df, 
    sentiment_df, 
    on=['Date', 'Ticker'], 
    how='left'
)

master_df['Sentiment_Score'] = master_df['Sentiment_Score'].fillna(0.0)

print("\nSentiment Score Distribution:")
print(master_df['Sentiment_Score'].value_counts(normalize=True).head())


master_df = master_df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

print("\nDataset Preview:")
print(master_df.tail(10)) 
print(f"\nTotal rows ready for deep learning: {len(master_df)}")

output_file = 'dataset_ready.parquet'
master_df.to_parquet(output_file, engine='pyarrow')
print(f"\nSuccessfully saved the master dataset to {output_file}")