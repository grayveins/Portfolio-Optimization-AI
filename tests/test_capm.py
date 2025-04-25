from src.core.expected_return import CapmCalculator

tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2022-01-01"
end_date = "2024-01-01"

capm = CapmCalculator(start_date, end_date)

print("Calculating CAPM expected returns...\n")

try:
    result = capm.calculate_expected_return(tickers)
    print("\nFinal CAPM expected returns:")
    print(result)
except Exception as e:
    print(f"\nError occurred: {e}")
