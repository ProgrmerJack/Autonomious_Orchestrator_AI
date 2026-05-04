"""Find what query format gets YF v1 search to return quotes."""

import urllib.request, urllib.parse, json


def search_yf(q):
    params = urllib.parse.urlencode({"q": q, "quotesCount": 5, "newsCount": 3})
    req = urllib.request.Request(
        f"https://query1.finance.yahoo.com/v1/finance/search?{params}",
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        quotes = data.get("quotes") or []
        return [
            (
                q2.get("symbol"),
                q2.get("quoteType"),
                q2.get("longname") or q2.get("shortname"),
            )
            for q2 in quotes[:3]
        ]
    except Exception as e:
        return [f"error: {e}"]


tests = [
    "NVIDIA",
    "NVDA",
    "nvidia quarterly earnings",
    "NVIDIA earnings growth",
    "NVIDIA Corporation",
    "nvidia stock",
    "AAPL",
    "Apple stock",
    "microsoft",
]
for t in tests:
    result = search_yf(t)
    print(f"  {repr(t)}: {result}")
