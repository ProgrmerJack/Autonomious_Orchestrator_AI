"""Test raw API calls with detailed error reporting."""
import json
import urllib.request
import urllib.parse
import urllib.error

def get_with_details(url, headers):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
            return resp.status, data[:800]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:400]
    except Exception as ex:
        return -1, str(ex)

# Test 1: Yahoo Finance v10 quoteSummary with proper browser-like headers
url1 = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/NVDA?modules=price,financialData,defaultKeyStatistics"
code1, body1 = get_with_details(url1, {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
})
print(f"YF v10 status: {code1}")
print(f"YF v10 body: {body1[:400]}")
print()

# Test 2: Yahoo Finance v7 (older but more permissive)
url2 = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=NVDA"
code2, body2 = get_with_details(url2, {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
})
print(f"YF v7 status: {code2}")
print(f"YF v7 body: {body2[:400]}")
print()

# Test 3: Yahoo Finance v8 (commonly used)
url3 = "https://query1.finance.yahoo.com/v8/finance/chart/NVDA?interval=1d&range=1d"
code3, body3 = get_with_details(url3, {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
})
print(f"YF v8 chart status: {code3}")
print(f"YF v8 chart body: {body3[:400]}")
print()

# Test 4: SEC EDGAR EFTS with SEC-required user agent
url4 = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode({
    "q": "NVIDIA", "forms": "10-K", "dateRange": "custom", "startdt": "2023-01-01"
})
code4, body4 = get_with_details(url4, {
    "Accept": "application/json",
    "User-Agent": "research-bot/1.0 (research@example.com)",
})
print(f"EDGAR EFTS status: {code4}")
print(f"EDGAR EFTS body: {body4[:600]}")
print()

# Test 5: SEC EDGAR search API (different endpoint)
url5 = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode({
    "q": "NVIDIA", "forms": "10-K"
})
code5, body5 = get_with_details(url5, {
    "User-Agent": "agentos/1.0 (research@example.com)",
    "Accept": "application/json",
})
print(f"EDGAR EFTS v2 status: {code5}")
print(f"EDGAR EFTS v2 body: {body5[:400]}")

# Test 6: EDGAR company search JSON
url6 = "https://www.sec.gov/cgi-bin/browse-edgar?company=nvidia&CIK=&type=10-K&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom"
code6, body6 = get_with_details(url6, {
    "User-Agent": "agentos/1.0 (research@example.com)",
    "Accept": "application/json, application/atom+xml, */*",
})
print(f"EDGAR browse status: {code6}")
print(f"EDGAR browse body: {body6[:400]}")

# Test 7: EDGAR full text search
url7 = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode({
    "q": '"NVIDIA" "annual report"',
    "dateRange": "custom",
    "startdt": "2024-01-01",
    "enddt": "2024-12-31",
    "forms": "10-K",
})
code7, body7 = get_with_details(url7, {
    "User-Agent": "research-bot/1.0 (research@example.com)",
    "Accept": "*/*",
})
print(f"EDGAR FTS v3 status: {code7}")
print(f"EDGAR FTS v3 body: {body7[:600]}")
