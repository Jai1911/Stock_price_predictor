import os
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import finnhub
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from transformers import pipeline
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


mcp = FastMCP("StockAnalyst")

print("System: Loading Neural Networks and Data Schemas...")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None)
tft_model = TemporalFusionTransformer.load_from_checkpoint("tft_finance_model_2.ckpt")
tft_model.eval()


training_ds_template = torch.load(
        "training_ds.pkl", 
        weights_only=False
    )
FINNHUB_CLIENT = finnhub.Client(api_key="FIN_API_KEY")

@mcp.tool()
async def get_live_market_forecast(ticker: str) -> str:
    """
    Fetches real-time market data, VIX, and news sentiment to generate 
    a 24-hour stock price forecast using a hybrid Deep Learning model.
    """
    ticker = ticker.upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120) 

    try:
        print(f"Fetching data for {ticker}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        vix_data = yf.download('^VIX', start=start_date, end=end_date)

        
        if stock_data.empty or vix_data.empty:
            return f"Error: Could not find market data for {ticker}."

        news = FINNHUB_CLIENT.company_news(ticker, _from=(end_date - timedelta(days=1)).strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))
        headlines = [n['headline'] for n in news[:5]]
        
        sentiment_scores = []
        for h in headlines:
            res = sentiment_analyzer(h[:512])[0]
            scores = {r['label']: r['score'] for r in res}
            sentiment_scores.append(scores.get('positive', 0) - scores.get('negative', 0))
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        live_df = pd.merge(stock_data[['Close', 'Volume']], vix_data[['Close']], 
                           left_index=True, right_index=True, how='left')
        live_df.columns = ['Close', 'Volume', 'VIX_Close']
        live_df = live_df.reset_index()

        live_df['Daily_Return'] = live_df['Close'].pct_change()
        live_df['Vol_14D'] = live_df['Daily_Return'].rolling(window=14).std()

        delta = live_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        live_df['RSI_14'] = 100 - (100 / (1 + gain/loss))
        ema12 = live_df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = live_df['Close'].ewm(span=26, adjust=False).mean()
        live_df['MACD'] = ema12 - ema26

        direction = np.sign(live_df['Close'].diff()).fillna(0)
        live_df['OBV'] = (direction * live_df['Volume']).cumsum()
        sma50 = live_df['Close'].rolling(50).mean()
        live_df['Dist_From_50SMA'] = (live_df['Close'] - sma50) / sma50
        full_df = live_df.dropna().copy()
        full_df['Target_Next_Day_Return'] = full_df['Daily_Return'].shift(-1).fillna(0.0)
        full_df['Ticker'] = ticker
        full_df['DayOfWeek'] = full_df['Date'].dt.dayofweek.astype(str)
        full_df['Sentiment_Score'] = avg_sentiment 
        
        full_df['time_idx'] = range(len(full_df))
        tomorrow = full_df.iloc[-1].copy()
        tomorrow['time_idx'] += 1
        tomorrow['Date'] += timedelta(days=1)
        tomorrow['DayOfWeek'] = str(tomorrow['Date'].dayofweek)


        full_df.loc[len(full_df)] = tomorrow

        inference_ds = TimeSeriesDataSet.from_dataset(
            training_ds_template, 
            full_df, 
            predict=True, 
            stop_randomization=True
        )
        
        inference_loader = inference_ds.to_dataloader(train=False, batch_size=1)
        
        with torch.no_grad():
            quantiles = tft_model.predict(inference_loader, mode="quantiles")
            
            p10 = quantiles[0, 0, 0].item()
            p50 = quantiles[0, 0, 1].item()
            p90 = quantiles[0, 0, 2].item()
            
        return (
            f"### Statistical Forecast for {ticker}\n"
            f"- **Predicted Next-Day Return:** {p50*100:.2f}%\n"
            f"- **Risk Bounds:** [{p10*100:.2f}%, {p90*100:.2f}%]\n"
            f"- **Market VIX:** {full_df['VIX_Close'].iloc[-2]:.2f}\n"
            f"- **News Sentiment:** {avg_sentiment:.2f}\n\n"
            f"**Key Headlines:**\n"
            + "\n".join([f"- {h}" for h in headlines])
        )

    except Exception as e:
        return f"Error executing inference for {ticker}: {str(e)}"

if __name__ == "__main__":
    mcp.run()