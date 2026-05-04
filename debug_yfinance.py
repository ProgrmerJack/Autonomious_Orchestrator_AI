"""Verify yfinance installation and what data it provides."""
import yfinance as yf
print("yfinance version:", yf.__version__)

nvda = yf.Ticker("NVDA")
info = nvda.info
print("\nAll yfinance info keys:", sorted(info.keys()))
print("\nKey financials:")
for key in ["regularMarketPrice","marketCap","trailingPE","forwardPE","priceToBook",
            "enterpriseToEbitda","shortPercentOfFloat","52WeekChange",
            "targetMeanPrice","numberOfAnalystOpinions","recommendationKey",
            "revenueGrowth","earningsGrowth","returnOnEquity",
            "totalRevenue","netIncomeToCommon","trailingEps","forwardEps",
            "dividendYield","beta","freeCashflow","currentPrice",
            "fiftyTwoWeekHigh","fiftyTwoWeekLow","longName","sector","industry"]:
    val = info.get(key)
    if val is not None:
        print(f"  {key}: {val}")
