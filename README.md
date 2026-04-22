# AI Quant Engine & Stock Analyst Model

This directory contains the machine learning models, API servers, and related data for predicting stock movements and generating analyst reports.

## Overview

The system uses a hybrid approach:
1. **Deep Learning Forecasting:** A PyTorch-based Temporal Fusion Transformer (TFT) model predicts the next-day stock returns using historical prices, VIX data, and engineered features.
2. **Sentiment Analysis:** Uses the `ProsusAI/finbert` model to calculate net sentiment from recent news headlines via Finnhub.
3. **LLM Synthesis:** Uses Google's Gemini to synthesize the quantitative data and sentiment into a human-readable analyst report.

## Files & Components

*   **`api.py`**: A FastAPI application providing the `/forecast/{ticker}` endpoint. It orchestrates the inference and calls Gemini for the final briefing.
*   **`server.py`**: An MCP (Model Context Protocol) server (`StockAnalyst`) exposing the `get_live_market_forecast` tool. It handles data fetching, feature engineering, and TFT model inference.
*   **`tft_finance_model_2.ckpt`**: The trained PyTorch Temporal Fusion Transformer model checkpoint.
*   **`training_ds.pkl`**: The PyTorch Forecasting `TimeSeriesDataSet` used to normalize incoming live data exactly like the training data.
*   **`dataset_ready.parquet`**: Prepared historical dataset used for training and validation.
*   **`requirements.txt`**: Python dependencies required to run the models and servers.

## Getting Started

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Ensure you have your Finnhub and Gemini API keys configured.
3.  To run the FastAPI server:
    ```bash
    fastapi dev api.py
    ```
4.  To run the MCP server:
    ```bash
    python server.py
    ```

5. I have added a test.py file to try it out locally.    
