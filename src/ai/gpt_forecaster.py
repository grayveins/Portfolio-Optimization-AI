import openai
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class GPTForecaster:
    """
    Uses OpenAI's GPT model to generate market outlooks (bullish/bearish + confidence).
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model

    def generate_forecast(self, ticker: str) -> Dict[str, float]:
        """
        Generates a market forecast using GPT.

        Returns:
        - Dict with expected return estimate and confidence score
        """
        system_prompt = (
            "You are a financial analyst. Given a stock ticker, generate a brief expected return forecast "
            "(annualized % return) and a confidence level between 0 and 100. Use real market patterns, earnings context, "
            "and realistic analysis. Respond in JSON format like: {'expected_return': 0.07, 'confidence': 82}"
        )

        user_prompt = f"Forecast the outlook for {ticker} over the next year."

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            forecast_text = response.choices[0].message.content
            forecast_data = eval(forecast_text)  # Use json.loads() if strict JSON
            return forecast_data

        except Exception as e:
            print(f"Error generating forecast for {ticker}: {e}")
            return {"expected_return": 0.05, "confidence": 50}  # fallback

    def batch_forecast(self, tickers: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Forecasts expected returns for a batch of tickers.

        Returns:
        - Dict mapping each ticker to its forecast dict
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.generate_forecast(ticker)
        return results
