import asyncio
from server import get_live_market_forecast

async def test():
    # This calls your tool directly without touching the Gemini API
    result = await get_live_market_forecast("NVDA")
    print(result)

if __name__ == "__main__":
    asyncio.run(test())