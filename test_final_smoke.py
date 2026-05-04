"""Final end-to-end smoke test: verify all three real data sources work."""
from agentos_orchestrator.research.deep_research import DeepResearchEngine

e = DeepResearchEngine()

print("=" * 70)
print("TEST 1: Yahoo Finance fundamentals via yfinance (NVDA)")
print("=" * 70)
abstract = e._fetch_yf_fundamentals_abstract("NVDA", "NVIDIA")
if abstract:
    for part in abstract.split(" | "):
        print(f"  {part}")
else:
    print("  EMPTY - FAIL")

print()
print("=" * 70)
print("TEST 2: SEC XBRL financial facts (NVDA via company_tickers.json)")
print("=" * 70)
sec_source = e._fetch_sec_company_facts("NVDA")
if sec_source:
    print(f"  Title: {sec_source.title}")
    print(f"  Score: {sec_source.score}")
    print(f"  URL: {sec_source.url}")
    print("  Abstract:")
    for part in sec_source.abstract.split(" | "):
        print(f"    {part}")
else:
    print("  None returned - FAIL")

print()
print("=" * 70)
print("TEST 3: _search_financial_portals('NVIDIA stock valuation')")
print("=" * 70)
fp = e._search_financial_portals("NVIDIA stock valuation analysis", limit=5)
print(f"  Returned {len(fp)} sources")
for s in fp[:5]:
    print(f"  [{s.score:.0f}] {s.title[:60]}")
    print(f"        {s.abstract[:100]}")

print()
print("=" * 70)
print("TEST 4: _search_sec_edgar('NVIDIA earnings revenue')")
print("=" * 70)
se = e._search_sec_edgar("NVIDIA earnings revenue", limit=5)
print(f"  Returned {len(se)} sources")
for s in se[:5]:
    print(f"  [{s.score:.0f}] {s.title[:60]}")
    print(f"        {s.url[:80]}")
    print(f"        {s.abstract[:100]}")

print()
print("=" * 70)
print("TEST 5: Provider diagnostics summary")
print("=" * 70)
for d in e.provider_diagnostics:
    print(f"  [{d['provider']}] {d['status']}: {d['detail']}")
