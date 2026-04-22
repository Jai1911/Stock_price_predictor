import finnhub
import pandas as pd
import time
from datetime import datetime, timedelta
from transformers import pipeline

FINNHUB_API_KEY = "FIN_API_KEY"
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

tickers = ['AAPL', 'NVDA', 'AMZN', 'MSFT', 'GOOGL']
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

print("Loading FinBERT Model... (This may take a moment)")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None)

all_news_data = []

print(f"Fetching 1 year of news from {start_str} to {end_str}...")
for ticker in tickers:
    print(f"Pulling {ticker}...")
    try:
        news = finnhub_client.company_news(ticker, _from=start_str, to=end_str)
        for article in news:
            all_news_data.append({
                'Ticker': ticker,
                'Datetime_UNIX': article['datetime'],
                'Headline': article['headline'],
                'Summary': article['summary']
            })
        
        time.sleep(1) 
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

df_news = pd.DataFrame(all_news_data)

print("Aligning timestamps to trading days...")
df_news['Datetime_UTC'] = pd.to_datetime(df_news['Datetime_UNIX'], unit='s', utc=True)
df_news['Datetime_EST'] = df_news['Datetime_UTC'].dt.tz_convert('America/New_York')
df_news['Shifted_Time'] = df_news['Datetime_EST'] + pd.Timedelta(hours=8)

df_news['Target_Trading_Date'] = df_news['Shifted_Time'].dt.normalize()

df_news['Target_Trading_Date'] = df_news['Target_Trading_Date'] + pd.offsets.BDay(0)
df_news['Target_Trading_Date'] = df_news['Target_Trading_Date'].dt.tz_localize(None)

print("Running FinBERT sentiment analysis on headlines...")
df_news['Full_Text'] = df_news['Headline'] + ". " + df_news['Summary']

def get_sentiment_score(text):
    if not text or len(text) < 5:
        return 0.0
    try:
        result = sentiment_analyzer(text[:512])[0]
        scores = {res['label']: res['score'] for res in result}
        return scores.get('positive', 0) - scores.get('negative', 0)
    except:
        return 0.0
df_news['Sentiment_Score'] = df_news['Full_Text'].apply(get_sentiment_score)

print("Aggregating daily sentiment...")
daily_sentiment = df_news.groupby(['Target_Trading_Date', 'Ticker'])['Sentiment_Score'].mean().reset_index()

daily_sentiment.rename(columns={'Target_Trading_Date': 'Date'}, inplace=True)

print(daily_sentiment.head(10))

output_file = 'sentiment_1yr_finbert.parquet'
daily_sentiment.to_parquet(output_file, engine='pyarrow')
print(f"\nSuccessfully saved sentiment data to {output_file}")