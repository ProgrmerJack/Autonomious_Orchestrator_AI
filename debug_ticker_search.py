"""Debug why _search_financial_portals isn't picking up NVDA from 'NVIDIA' query."""
import urllib.request, urllib.parse, json

def get_json(url):
    req = urllib.request.Request(url, headers={"Accept":"application/json","User-Agent":"Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  error: {e}")
        return {}

# What does YF v1 search return for NVIDIA query?
yf_params = urllib.parse.urlencode({
    "q": "NVIDIA stock valuation analysis",
    "quotesCount": 12, "newsCount": 5, "enableNavLinks": "true",
    "enableEnhancedTrivialQuery": "true",
})
data = get_json(f"https://query1.finance.yahoo.com/v1/finance/search?{yf_params}")
print("YF search quotes:")
for item in (data.get("quotes") or [])[:10]:
    print(f"  symbol={item.get('symbol')} quoteType={item.get('quoteType')} name={item.get('longname','') or item.get('shortname','')}")

print("\nYF search news (first 3):")
for item in (data.get("news") or [])[:3]:
    print(f"  title={item.get('title','')[:60]} url={item.get('link','')[:60]}")

# Also test _extract_ticker_candidates for 'NVIDIA earnings revenue'
import sys
sys.path.insert(0, 'c:\\Users\\Jack0\\Autonomious_Orchestrator_AI')
from agentos_orchestrator.research.deep_research import _extract_ticker_candidates

tests = [
    "NVIDIA stock valuation analysis",
    "NVIDIA earnings revenue",
    "NVDA earnings",
    "Apple AAPL stock",
    "Microsoft Azure growth",
    "NVIDIA quarterly earnings report",
]
print("\n_extract_ticker_candidates results:")
for t in tests:
    result = _extract_ticker_candidates(t)
    print(f"  {repr(t[:40])} -> {result}")
