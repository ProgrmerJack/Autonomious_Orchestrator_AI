import json, urllib.parse
from agentos_orchestrator.research.deep_research import DeepResearchEngine

e = DeepResearchEngine()

# --- Debug YF quoteSummary ---
abstract = e._fetch_yf_fundamentals_abstract("NVDA", "NVIDIA")
print("YF fundamentals (NVDA):", repr(abstract[:200]) if abstract else "EMPTY")

# Manually inspect YF v10 raw payload
payload_v10 = e._get_json(
    "https://query1.finance.yahoo.com/v10/finance/quoteSummary/NVDA"
    "?modules=price,financialData,defaultKeyStatistics"
)
print("YF v10 top keys:", list(payload_v10.keys()))
qs = payload_v10.get("quoteSummary") or {}
print("quoteSummary keys:", list(qs.keys()))
result_list = qs.get("result") or []
print("result list len:", len(result_list))
if result_list:
    data = result_list[0]
    print("data keys:", list(data.keys()))
    price_d = data.get("price") or {}
    print("price.regularMarketPrice:", price_d.get("regularMarketPrice"))
else:
    err = qs.get("error") or payload_v10.get("error")
    print("error:", json.dumps(err, indent=2)[:400] if err else "no error field")
    print("raw (first 800):", json.dumps(payload_v10, indent=2)[:800])

print()

# --- Debug SEC EDGAR EFTS ---
params = urllib.parse.urlencode({
    "q": "NVIDIA",
    "forms": "10-K,10-Q,8-K",
    "dateRange": "custom",
    "startdt": "2023-01-01",
})
payload = e._get_sec_json(f"https://efts.sec.gov/LATEST/search-index?{params}")
hits = (payload.get("hits") or {}).get("hits") or []
print("EDGAR hits:", len(hits))
if hits:
    src = hits[0].get("_source", {})
    print(json.dumps({k: src[k] for k in list(src.keys())[:10]}, indent=2)[:600])
else:
    print("EDGAR raw (first 600):", json.dumps(payload, indent=2)[:600])

# Also try the EDGAR search API (not EFTS)
p2 = e._get_sec_json(
    "https://efts.sec.gov/LATEST/search-index?q=NVIDIA&forms=10-K&dateRange=custom&startdt=2023-01-01&hits.hits._source.period_of_report=2024"
)
print("EDGAR v2 hits:", len(((p2.get("hits") or {}).get("hits")) or []))

# Try the regular EDGAR search
p3 = e._get_json(
    "https://efts.sec.gov/LATEST/search-index?q=NVIDIA&forms=10-K"
)
print("EDGAR v3 (via _get_json):", list(p3.keys()) if p3 else "empty")
hits3 = (p3.get("hits") or {}).get("hits") or []
print("EDGAR v3 hits:", len(hits3))
