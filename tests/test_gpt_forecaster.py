import os
from src.ai.gpt_forecaster import GPTForecaster
from dotenv import load_dotenv
from pathlib import Path

# Force-load the .env file from the project root directory
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

def test_gpt_forecaster():
    # Ensure the OpenAI key is loaded
    assert os.getenv("OPENAI_API_KEY") is not None, " OPENAI_API_KEY not found. Check your .env."

    # Initialize the forecaster
    gpt = GPTForecaster(model="gpt-3.5-turbo")

    # Test single ticker
    forecast = gpt.generate_forecast("AAPL")
    assert "expected_return" in forecast and "confidence" in forecast, "Missing keys in forecast result"
    print(f"AAPL forecast: {forecast}")

    # Test batch forecast
    batch = gpt.batch_forecast(["MSFT", "GOOGL"])
    assert isinstance(batch, dict), "Batch result should be a dictionary"

    for ticker, result in batch.items():
        assert "expected_return" in result and "confidence" in result, f"Missing keys for {ticker}"
    print("Batch forecast:")
    print(batch)


if __name__ == "__main__":
    test_gpt_forecaster()
