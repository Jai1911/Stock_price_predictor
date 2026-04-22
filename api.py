from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from transformers import pipeline
from model.server import get_live_market_forecast
from google import genai
import os

print("Loading Deep Learning Models into Memory...")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None)
tft_model = TemporalFusionTransformer.load_from_checkpoint("tft_finance_model_2.ckpt")
tft_model.eval()

with torch.serialization.safe_globals([TimeSeriesDataSet]):
    training_ds_template = torch.load("training_ds.pkl", weights_only=False)

app = FastAPI(title="AI Quant Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = "GEM_API"
client = genai.Client(api_key=GEMINI_API_KEY)

@app.get("/forecast/{ticker}")
async def get_forecast(ticker: str):
    try:
        raw_quant_output = await get_live_market_forecast(
            ticker
        )
        
        if "Error" in raw_quant_output:
            raise HTTPException(status_code=400, detail=raw_quant_output)

        prompt = (
            f"You are a hedge fund analyst. I have run my proprietary model on {ticker}. "
            f"Here are the results:\n\n{raw_quant_output}\n\n"
            f"Write a concise briefing explaining the forecast and news drivers."
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        return {
            "ticker": ticker.upper(),
            "raw_data": raw_quant_output,
            "analyst_report": response.text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))