"""Test data.sec.gov APIs and check for yfinance."""
import json
import urllib.request
import urllib.parse
import urllib.error

def get_with_details(url, headers):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
            return resp.status, data
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:400]
    except Exception as ex:
        return -1, str(ex)

SEC_UA = "agentos/1.0 (research@example.com)"

# Test 1: Yahoo Finance v8 chart for more details
url1 = "https://query1.finance.yahoo.com/v8/finance/chart/NVDA?interval=1d&range=1d&includeTimestamps=false"
code1, body1 = get_with_details(url1, {"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
print(f"YF v8 chart status: {code1}")
if code1 == 200:
    data = json.loads(body1)
    meta = (data.get("chart", {}).get("result") or [{}])[0].get("meta", {})
    print("Meta keys:", list(meta.keys()))
    print("regularMarketPrice:", meta.get("regularMarketPrice"))
    print("52W High:", meta.get("fiftyTwoWeekHigh"))
    print("52W Low:", meta.get("fiftyTwoWeekLow"))
    print("Market cap present:", "marketCap" in meta)
print()

# Test 2: Yahoo Finance v8 with modules (sometimes works)
url2 = "https://query1.finance.yahoo.com/v8/finance/chart/NVDA?interval=1d&range=1y&includeTimestamps=false"
code2, body2 = get_with_details(url2, {"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
print(f"YF v8 1y status: {code2}")
if code2 == 200:
    data2 = json.loads(body2)
    meta2 = (data2.get("chart", {}).get("result") or [{}])[0].get("meta", {})
    print("All meta fields:", json.dumps({k:v for k,v in list(meta2.items())[:20]}, indent=2)[:600])
print()

# Test 3: data.sec.gov/submissions (company facts)
url3 = "https://data.sec.gov/submissions/CIK0001045810.json"  # NVIDIA CIK
code3, body3 = get_with_details(url3, {"User-Agent": SEC_UA, "Accept": "application/json"})
print(f"SEC submissions status: {code3}")
if code3 == 200:
    data3 = json.loads(body3)
    print("name:", data3.get("name"))
    print("sic:", data3.get("sic"))
    print("sicDesc:", data3.get("sicDescription"))
    print("exchanges:", data3.get("exchanges"))
    recent = data3.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    print("Recent 10-K:", [(f, d) for f, d in zip(forms[:30], dates[:30]) if f == "10-K"][:3])
print()

# Test 4: data.sec.gov/api/xbrl/companyfacts
url4 = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json"
code4, body4 = get_with_details(url4, {"User-Agent": SEC_UA, "Accept": "application/json"})
print(f"SEC XBRL facts status: {code4}")
if code4 == 200:
    data4 = json.loads(body4)
    gaap = (data4.get("facts") or {}).get("us-gaap") or {}
    print("GAAP concepts count:", len(gaap))
    rev = gaap.get("Revenues", {})
    if rev:
        entries = rev.get("units", {}).get("USD", [])
        annual = sorted([e for e in entries if e.get("form") == "10-K" and e.get("val")], 
                       key=lambda e: e.get("end",""), reverse=True)
        print("Revenue entries (10-K, latest 3):", [(e.get("end"), e.get("val")) for e in annual[:3]])
    else:
        alt = gaap.get("RevenueFromContractWithCustomerExcludingAssessedTax", {})
        if alt:
            entries = alt.get("units", {}).get("USD", [])
            annual = sorted([e for e in entries if e.get("form") == "10-K" and e.get("val")],
                           key=lambda e: e.get("end",""), reverse=True)
            print("Revenue (alt concept, 10-K, latest 3):", [(e.get("end"), e.get("val")) for e in annual[:3]])
else:
    print("XBRL body:", body4[:300])
print()

# Test 5: SEC company tickers list (maps tickers to CIK)
url5 = "https://www.sec.gov/files/company_tickers.json"
code5, body5 = get_with_details(url5, {"User-Agent": SEC_UA, "Accept": "application/json"})
print(f"SEC company tickers status: {code5}")
if code5 == 200:
    data5 = json.loads(body5)
    # Find NVDA
    for k, v in list(data5.items())[:5]:
        print("  Sample:", k, "->", v)
    # Search for NVDA
    for k, v in data5.items():
        if v.get("ticker") == "NVDA":
            print("  NVDA entry:", v)
            break
print()

# Test 6: check if yfinance is installed
try:
    import yfinance as yf
    print("yfinance installed:", yf.__version__)
    nvda = yf.Ticker("NVDA")
    info = nvda.info
    print("yfinance NVDA keys:", list(info.keys())[:10])
    print("yfinance NVDA price:", info.get("regularMarketPrice") or info.get("currentPrice"))
    print("yfinance NVDA marketCap:", info.get("marketCap"))
    print("yfinance NVDA forwardPE:", info.get("forwardPE"))
except ImportError:
    print("yfinance NOT installed")
except Exception as ex:
    print("yfinance error:", ex)
