"""Test browse-edgar Atom XML parsing for filing list."""
import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET

SEC_UA = "agentos/1.0 (research@example.com)"

def get_sec(url):
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:400]
    except Exception as ex:
        return -1, str(ex)

# Test 1: browse-edgar Atom for NVDA (CIK known = 1045810)
url1 = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=1045810&type=10-K&dateb=&owner=include&count=10&output=atom"
code1, body1 = get_sec(url1)
print(f"browse-edgar CIK atom status: {code1}")
if code1 == 200:
    try:
        root = ET.fromstring(body1)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        # Get company-info
        ci = root.find("company-info")
        if ci is not None:
            print("  company:", ci.findtext("conformed-name", ""))
            print("  SIC:", ci.findtext("assigned-sic-desc", ""))
            print("  state:", ci.findtext("state-of-incorporation", ""))
        entries = root.findall("atom:entry", ns)
        print(f"  {len(entries)} filing entries")
        for entry in entries[:3]:
            title = entry.findtext("atom:title", "", ns)
            link_el = entry.find("atom:link", ns)
            href = link_el.get("href", "") if link_el is not None else ""
            updated = entry.findtext("atom:updated", "", ns)
            cat_el = entry.find("atom:category", ns)
            cat = cat_el.get("term", "") if cat_el is not None else ""
            print(f"  [{updated[:10]}] {title[:60]} -> {href[:80]}")
            print(f"           category: {cat}")
    except ET.ParseError as e:
        print("  XML parse error:", e)
        print("  Body:", body1[:400])
print()

# Test 2: browse-edgar company name search
url2 = "https://www.sec.gov/cgi-bin/browse-edgar?company=nvidia&CIK=&type=10-K&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom"
code2, body2 = get_sec(url2)
print(f"browse-edgar name search status: {code2}")
if code2 == 200:
    try:
        root2 = ET.fromstring(body2)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries2 = root2.findall("atom:entry", ns)
        print(f"  {len(entries2)} results")
        for entry in entries2[:5]:
            title = entry.findtext("atom:title", "", ns)
            link_el = entry.find("atom:link", ns)
            href = link_el.get("href", "") if link_el is not None else ""
            print(f"  {title[:80]} -> {href[:80]}")
    except ET.ParseError as e:
        print("  XML error:", e)
print()

# Test 3: data.sec.gov submissions for NVDA with proper throttle
import time, json
time.sleep(1.0)
url3 = "https://data.sec.gov/submissions/CIK0001045810.json"
code3, body3 = get_sec(url3)
print(f"data.sec.gov/submissions status: {code3}")
if code3 == 200:
    d = json.loads(body3)
    print("  name:", d.get("name"))
    print("  sic:", d.get("sicDescription"))
    recent = d.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    print("  Recent 10-K filings:")
    for f, d2, a in zip(forms, dates, accessions):
        if f == "10-K":
            print(f"    {d2} {a}")
            break
print()

# Test 4: data.sec.gov XBRL facts with proper throttle
time.sleep(1.0)
url4 = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json"
code4, body4 = get_sec(url4)
print(f"data.sec.gov/xbrl status: {code4}")
if code4 == 200:
    d4 = json.loads(body4)
    gaap = d4.get("facts", {}).get("us-gaap", {})
    print("  GAAP concepts:", len(gaap))
    for rev_key in ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]:
        if rev_key in gaap:
            entries = gaap[rev_key].get("units", {}).get("USD", [])
            annual = sorted([e for e in entries if e.get("form") == "10-K" and e.get("val")],
                           key=lambda e: e.get("end",""), reverse=True)[:3]
            print(f"  {rev_key} (10-K, latest 3): {[(e['end'], e['val']) for e in annual]}")
            break
else:
    print("  Error:", body4[:200])
