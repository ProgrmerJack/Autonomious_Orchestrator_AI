from __future__ import annotations

import asyncio
import html
import json
import os
import re
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from .models import ResearchSource, extract_ticker_candidates as _extract_ticker_candidates
from .provider_policy import provider_order as _provider_order_policy


class ProviderSearchMixin:
    def _search_semantic_scholar(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": str(limit or self.limit_per_provider),
                "fields": ",".join(
                    [
                        "title",
                        "abstract",
                        "authors",
                        "year",
                        "url",
                        "citationCount",
                        "openAccessPdf",
                    ]
                ),
            }
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        payload = self._get_json(url)
        sources: list[ResearchSource] = []
        for item in payload.get("data", []):
            title = html.unescape(str(item.get("title") or "").strip())
            if not title:
                continue
            open_pdf = item.get("openAccessPdf") or {}
            source_url = str(open_pdf.get("url") or item.get("url") or "")
            citation_count = int(item.get("citationCount") or 0)
            sources.append(
                ResearchSource(
                    provider="semantic-scholar",
                    title=title,
                    url=source_url,
                    year=item.get("year"),
                    authors=[
                        str(author.get("name"))
                        for author in item.get("authors", [])[:6]
                        if author.get("name")
                    ],
                    abstract=str(item.get("abstract") or ""),
                    citation_count=citation_count,
                    score=float(citation_count),
                )
            )
        self._record_provider_diagnostic(
            "semantic-scholar",
            "ok" if sources else "empty",
            f"returned {len(sources)} records",
        )
        return sources

    async def _search_semantic_scholar_async(
        self,
        query: str,
        limit: int | None = None,
        client: Any | None = None,
    ) -> list[ResearchSource]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": str(limit or self.limit_per_provider),
                "fields": ",".join(
                    [
                        "title",
                        "abstract",
                        "authors",
                        "year",
                        "url",
                        "citationCount",
                        "openAccessPdf",
                    ]
                ),
            }
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        payload = await self._get_json_async(url, client=client)
        sources: list[ResearchSource] = []
        for item in payload.get("data", []):
            title = html.unescape(str(item.get("title") or "").strip())
            if not title:
                continue
            open_pdf = item.get("openAccessPdf") or {}
            source_url = str(open_pdf.get("url") or item.get("url") or "")
            citation_count = int(item.get("citationCount") or 0)
            sources.append(
                ResearchSource(
                    provider="semantic-scholar",
                    title=title,
                    url=source_url,
                    year=item.get("year"),
                    authors=[
                        str(author.get("name"))
                        for author in item.get("authors", [])[:6]
                        if author.get("name")
                    ],
                    abstract=str(item.get("abstract") or ""),
                    citation_count=citation_count,
                    score=float(citation_count),
                )
            )
        self._record_provider_diagnostic(
            "semantic-scholar",
            "ok" if sources else "empty",
            f"returned {len(sources)} records",
        )
        return sources

    def _search_crossref(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        params = urllib.parse.urlencode(
            {
                "query.bibliographic": query,
                "rows": str(limit or self.limit_per_provider),
                "sort": "relevance",
                "order": "desc",
            }
        )
        payload = self._get_json(f"https://api.crossref.org/works?{params}")
        message = payload.get("message") if isinstance(payload, dict) else {}
        items = message.get("items") if isinstance(message, dict) else []
        sources: list[ResearchSource] = []
        for item in items or []:
            title_list = item.get("title") or []
            title = html.unescape(str(title_list[0] if title_list else "").strip())
            if not title:
                continue
            doi = str(item.get("DOI") or "").strip()
            source_url = str(
                item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            )
            created = item.get("created") or {}
            date_parts = created.get("date-parts") or []
            year = None
            if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                try:
                    year = int(date_parts[0][0])
                except (TypeError, ValueError):
                    year = None
            authors = []
            for author in item.get("author", [])[:6]:
                given = str(author.get("given") or "").strip()
                family = str(author.get("family") or "").strip()
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            citation_count = int(item.get("is-referenced-by-count") or 0)
            sources.append(
                ResearchSource(
                    provider="crossref",
                    title=title,
                    url=source_url,
                    year=year,
                    authors=authors,
                    abstract=str(item.get("abstract") or ""),
                    citation_count=citation_count,
                    score=float(citation_count),
                )
            )
        self._record_provider_diagnostic(
            "crossref",
            "ok" if sources else "empty",
            f"returned {len(sources)} records",
        )
        return sources

    async def _search_crossref_async(
        self,
        query: str,
        limit: int | None = None,
        client: Any | None = None,
    ) -> list[ResearchSource]:
        params = urllib.parse.urlencode(
            {
                "query.bibliographic": query,
                "rows": str(limit or self.limit_per_provider),
                "sort": "relevance",
                "order": "desc",
            }
        )
        payload = await self._get_json_async(
            f"https://api.crossref.org/works?{params}",
            client=client,
        )
        message = payload.get("message") if isinstance(payload, dict) else {}
        items = message.get("items") if isinstance(message, dict) else []
        sources: list[ResearchSource] = []
        for item in items or []:
            title_list = item.get("title") or []
            title = html.unescape(str(title_list[0] if title_list else "").strip())
            if not title:
                continue
            doi = str(item.get("DOI") or "").strip()
            source_url = str(
                item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            )
            created = item.get("created") or {}
            date_parts = created.get("date-parts") or []
            year = None
            if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                try:
                    year = int(date_parts[0][0])
                except (TypeError, ValueError):
                    year = None
            authors = []
            for author in item.get("author", [])[:6]:
                given = str(author.get("given") or "").strip()
                family = str(author.get("family") or "").strip()
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            citation_count = int(item.get("is-referenced-by-count") or 0)
            sources.append(
                ResearchSource(
                    provider="crossref",
                    title=title,
                    url=source_url,
                    year=year,
                    authors=authors,
                    abstract=str(item.get("abstract") or ""),
                    citation_count=citation_count,
                    score=float(citation_count),
                )
            )
        self._record_provider_diagnostic(
            "crossref",
            "ok" if sources else "empty",
            f"returned {len(sources)} records",
        )
        return sources

    def _get_sec_tickers(self) -> "dict[str, dict]":
        """Return {TICKER: {cik_str, ticker, title}} from SEC company_tickers.json.

        Cached at class level after first fetch — the file is ~1.5 MB and lists
        every ~15 000 public company registered with the SEC.  Used to resolve
        a ticker symbol (e.g. "NVDA") to the SEC's numeric CIK so we can call
        data.sec.gov/submissions and data.sec.gov/api/xbrl/companyfacts.
        """
        if self.__class__._sec_tickers_cache is not None:
            return self.__class__._sec_tickers_cache
        payload = self._get_sec_json("https://www.sec.gov/files/company_tickers.json")
        if not payload:
            return {}
        result: dict[str, dict] = {}
        for item in payload.values():
            ticker = str(item.get("ticker") or "").upper().strip()
            if ticker:
                result[ticker] = item
        self.__class__._sec_tickers_cache = result
        return result

    def _fetch_yf_chart_abstract(self, symbol: str, name: str) -> str:
        """Fallback: basic price + 52-week range from Yahoo Finance v8 chart API.

        The v8 chart endpoint works without authentication.  Returns a minimal
        but real abstract (price, 52W range, volume, exchange) for tickers where
        yfinance is unavailable.  Returns empty string on failure.
        """
        payload = self._get_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?interval=1d&range=1d"
        )
        meta = ((payload.get("chart") or {}).get("result") or [{}])[0].get("meta") or {}
        price = meta.get("regularMarketPrice")
        if not price:
            return ""
        high_52w = meta.get("fiftyTwoWeekHigh")
        low_52w = meta.get("fiftyTwoWeekLow")
        volume = meta.get("regularMarketVolume")
        exchange = meta.get("fullExchangeName") or meta.get("exchangeName") or ""
        long_name = meta.get("longName") or meta.get("shortName") or name

        parts: list[str] = [f"{long_name} ({symbol})"]
        if exchange:
            parts.append(exchange)
        parts.append(f"Price: ${price:.2f}")
        if high_52w and low_52w:
            parts.append(f"52W Range: ${low_52w:.2f}–${high_52w:.2f}")
        if volume:
            parts.append(f"Volume: {volume:,}")
        return " | ".join(parts)

    def _get_sec_json(self, url: str) -> dict[str, Any]:
        """Fetch JSON from SEC APIs (data.sec.gov, efts.sec.gov).

        SEC policy requires a descriptive User-Agent with contact info.
        Returns {} on any error — callers must handle gracefully.
        """
        try:
            return json.loads(self._get_sec_text(url, accept="application/json"))
        except json.JSONDecodeError:
            return {}

    def _get_sec_text(
        self,
        url: str,
        accept: str = "application/json",
    ) -> str:
        """Fetch SEC text content with retry/backoff for transient throttling."""
        sec_ua = os.environ.get(
            "SEC_USER_AGENT",
            "agentos-orchestrator/0.1 research-bot (set SEC_USER_AGENT)",
        )
        headers = {
            "Accept": accept,
            "User-Agent": sec_ua,
        }
        retry_delays = (0.0, 1.0, 2.5)

        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError:
            return ""

        # http2=True requires the optional 'h2' package; fall back to http/1.1
        # when it is absent so SEC EDGAR calls never crash with ImportError.
        try:
            import h2  # noqa: F401  # type: ignore[import-not-found]
            _http2 = True
        except ImportError:
            _http2 = False

        with httpx.Client(
            follow_redirects=True,
            http2=_http2,
            verify=True,
        ) as client:
            for delay in retry_delays:
                if delay > 0:
                    time.sleep(delay)
                try:
                    response = client.get(
                        url,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                except httpx.HTTPError:
                    continue
                if response.status_code == 200:
                    return response.text
                if response.status_code in {403, 429, 500, 502, 503, 504}:
                    continue
                break
        return ""

    def _fetch_yf_fundamentals_abstract(self, symbol: str, name: str) -> str:
        """Fetch REAL fundamental data using yfinance (handles Yahoo Finance auth).

        yfinance handles the Yahoo Finance crumb/cookie authentication that
        blocks raw urllib calls.  Returns a pipe-delimited string of live
        metrics a Wall Street analyst actually uses: P/E, market cap, analyst
        consensus and price targets, growth rates, ROE, FCF, short interest.

        Falls back to the v8 chart API (price + 52W range) if yfinance is
        not installed.  Returns empty string only if ALL data paths fail,
        so callers skip the source rather than emit a placeholder.
        """
        try:
            import yfinance as yf  # optional; falls back gracefully
        except ImportError:
            return self._fetch_yf_chart_abstract(symbol, name)

        try:
            info: dict[str, Any] = yf.Ticker(symbol).info or {}
        except Exception:
            return self._fetch_yf_chart_abstract(symbol, name)

        if not info:
            return self._fetch_yf_chart_abstract(symbol, name)

        long_name = str(info.get("longName") or info.get("shortName") or name)
        curr_price = info.get("currentPrice") or info.get("regularMarketPrice")
        mkt_cap_raw = info.get("marketCap")
        mkt_cap = f"${mkt_cap_raw / 1e9:.1f}B" if mkt_cap_raw else ""
        sector = str(info.get("sector") or "")
        industry = str(info.get("industry") or "")

        trailing_pe = info.get("trailingPE")
        fwd_pe = info.get("forwardPE")
        pb = info.get("priceToBook")
        ev_ebitda = info.get("enterpriseToEbitda")
        beta = info.get("beta")
        short_float_raw = info.get("shortPercentOfFloat")
        wk52_chg_raw = info.get("52WeekChange") or info.get("fiftyTwoWeekChangePercent")
        target_price = info.get("targetMeanPrice")
        n_analysts = info.get("numberOfAnalystOpinions")
        rec = str(info.get("recommendationKey") or "").upper().replace("_", " ")
        rev_growth_raw = info.get("revenueGrowth")
        earn_growth_raw = info.get("earningsGrowth")
        roe_raw = info.get("returnOnEquity")
        total_rev = info.get("totalRevenue")
        net_income = info.get("netIncomeToCommon")
        free_cf = info.get("freeCashflow")
        fwd_eps = info.get("forwardEps")
        trailing_eps = info.get("trailingEps")

        parts: list[str] = [f"{long_name} ({symbol})"]
        if sector:
            parts.append(f"{sector} — {industry}" if industry else sector)
        if curr_price:
            parts.append(f"Price: ${curr_price:.2f}")
        if mkt_cap:
            parts.append(f"Mkt Cap: {mkt_cap}")
        if trailing_pe:
            parts.append(f"Trailing P/E: {trailing_pe:.1f}x")
        if fwd_pe:
            parts.append(f"Fwd P/E: {fwd_pe:.1f}x")
        if ev_ebitda:
            parts.append(f"EV/EBITDA: {ev_ebitda:.1f}x")
        if pb:
            parts.append(f"P/B: {pb:.2f}x")
        if beta:
            parts.append(f"Beta: {beta:.2f}")
        if short_float_raw:
            parts.append(f"Short Float: {short_float_raw * 100:.1f}%")
        if wk52_chg_raw:
            parts.append(f"52W Chg: {wk52_chg_raw * 100:+.1f}%")
        if target_price and curr_price and curr_price > 0:
            upside_pct = (target_price / curr_price - 1) * 100
            analyst_note = (
                f"({upside_pct:+.1f}% upside, {int(n_analysts)} analysts)"
                if n_analysts
                else f"({upside_pct:+.1f}% upside)"
            )
            parts.append(f"Target: ${target_price:.2f} {analyst_note}")
        if rec:
            parts.append(f"Rating: {rec}")
        if total_rev:
            parts.append(f"Revenue: ${total_rev / 1e9:.1f}B")
        if rev_growth_raw:
            parts.append(f"Rev Growth: {rev_growth_raw * 100:+.1f}%")
        if net_income:
            parts.append(f"Net Income: ${net_income / 1e9:.1f}B")
        if earn_growth_raw:
            parts.append(f"EPS Growth: {earn_growth_raw * 100:+.1f}%")
        if roe_raw:
            parts.append(f"ROE: {roe_raw * 100:+.1f}%")
        if free_cf:
            parts.append(f"FCF: ${free_cf / 1e9:.1f}B")
        if fwd_eps:
            parts.append(f"Fwd EPS: ${fwd_eps:.2f}")
        if trailing_eps:
            parts.append(f"TTM EPS: ${trailing_eps:.2f}")

        result = " | ".join(p for p in parts if p)
        return result if len(parts) > 2 else self._fetch_yf_chart_abstract(symbol, name)

    def _fetch_sec_company_facts(self, ticker: str) -> "ResearchSource | None":
        """Fetch real audited financial data from SEC XBRL API.

        Pipeline:
        1. company_tickers.json → resolve ticker to CIK (cached class-level)
        2. data.sec.gov/submissions/CIK{n}.json → company name, SIC, exchange,
           most recent 10-K and 10-Q filing dates
        3. data.sec.gov/api/xbrl/companyfacts/CIK{n}.json → audited GAAP data:
           revenue, net income, diluted EPS from actual 10-K filings

        Returns a high-confidence (score=72) ResearchSource with real SEC data,
        or None if the ticker is not found or data is insufficient.
        """
        import time

        # Step 1: ticker → CIK via the cached SEC company tickers mapping
        tickers_map = self._get_sec_tickers()
        entry = tickers_map.get(ticker.upper().strip())
        if not entry:
            return None
        cik_int = entry.get("cik_str") or entry.get("cik")
        if not cik_int:
            return None
        cik_padded = str(int(cik_int)).zfill(10)
        company_title = str(entry.get("title") or ticker)

        # Step 2: Company metadata from data.sec.gov/submissions
        submissions = self._get_sec_json(
            f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        )
        if not submissions:
            return None

        company_name = str(submissions.get("name") or company_title)
        sic_desc = str(submissions.get("sicDescription") or "")
        exchanges = submissions.get("exchanges") or []
        exchange = ", ".join(str(e) for e in exchanges) if exchanges else ""

        recent = (submissions.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        accessions = recent.get("accessionNumber") or []

        most_recent_10k_date = ""
        most_recent_10q_date = ""
        for i, form in enumerate(forms):
            if form == "10-K" and not most_recent_10k_date and i < len(dates):
                most_recent_10k_date = dates[i]
            if form == "10-Q" and not most_recent_10q_date and i < len(dates):
                most_recent_10q_date = dates[i]
            if most_recent_10k_date and most_recent_10q_date:
                break

        # Step 3: XBRL financials — audited GAAP data from actual SEC filings.
        # SEC policy: max 10 req/sec.  A brief pause keeps us well under limit.
        time.sleep(0.15)
        facts_payload = self._get_sec_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
        )
        gaap = (facts_payload.get("facts") or {}).get("us-gaap") or {}

        def _latest_annual(concept_key: str) -> "tuple[float, str] | None":
            entries = (gaap.get(concept_key) or {}).get("units", {}).get("USD") or []
            annual = sorted(
                [
                    e
                    for e in entries
                    if e.get("form") == "10-K" and e.get("val") is not None
                ],
                key=lambda e: e.get("end", ""),
                reverse=True,
            )
            if annual:
                return float(annual[0]["val"]), str(annual[0].get("end", ""))[:7]
            return None

        def _yoy_growth(concept_key: str) -> "float | None":
            entries = (gaap.get(concept_key) or {}).get("units", {}).get("USD") or []
            annual = sorted(
                [
                    e
                    for e in entries
                    if e.get("form") == "10-K" and e.get("val") is not None
                ],
                key=lambda e: e.get("end", ""),
                reverse=True,
            )
            if len(annual) >= 2 and annual[1].get("val") and annual[1]["val"] != 0:
                return (
                    (float(annual[0]["val"]) - float(annual[1]["val"]))
                    / abs(float(annual[1]["val"]))
                    * 100
                )
            return None

        # Revenue: try common GAAP concepts in priority order
        rev_key = next(
            (
                k
                for k in (
                    "Revenues",
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet",
                    "SalesRevenueGoodsNet",
                    "RevenueFromContractWithCustomerIncludingAssessedTax",
                )
                if k in gaap
            ),
            None,
        )

        abstract_parts: list[str] = [f"{company_name} (SEC CIK: {str(int(cik_int))})"]
        if sic_desc:
            abstract_parts.append(f"Industry: {sic_desc}")
        if exchange:
            abstract_parts.append(f"Exchange: {exchange}")
        if most_recent_10k_date:
            abstract_parts.append(f"Latest 10-K: {most_recent_10k_date}")
        if most_recent_10q_date:
            abstract_parts.append(f"Latest 10-Q: {most_recent_10q_date}")

        has_financials = False
        if rev_key:
            rev_result = _latest_annual(rev_key)
            if rev_result:
                rev_val, rev_period = rev_result
                abstract_parts.append(f"Revenue ({rev_period}): ${rev_val / 1e9:.2f}B")
                rev_growth = _yoy_growth(rev_key)
                if rev_growth is not None:
                    abstract_parts.append(f"Revenue YoY: {rev_growth:+.1f}%")
                has_financials = True

        if "NetIncomeLoss" in gaap:
            ni_result = _latest_annual("NetIncomeLoss")
            if ni_result:
                ni_val, ni_period = ni_result
                abstract_parts.append(f"Net Income ({ni_period}): ${ni_val / 1e9:.2f}B")
                ni_growth = _yoy_growth("NetIncomeLoss")
                if ni_growth is not None:
                    abstract_parts.append(f"Net Income YoY: {ni_growth:+.1f}%")
                has_financials = True

        if "EarningsPerShareDiluted" in gaap:
            eps_result = _latest_annual("EarningsPerShareDiluted")
            if eps_result:
                eps_val, eps_period = eps_result
                abstract_parts.append(f"Diluted EPS ({eps_period}): ${eps_val:.2f}")

        if len(abstract_parts) < 3 and not has_financials:
            return None

        cik_int_str = str(int(cik_int))
        return ResearchSource(
            provider="sec-edgar",
            title=f"{company_name} — SEC XBRL Financial Facts",
            url=(
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={cik_int_str}"
                f"&type=10-K&dateb=&owner=include&count=10"
            ),
            year=datetime.now(UTC).year,
            abstract=" | ".join(p for p in abstract_parts if p),
            score=72.0,  # Audited primary-source data — highest confidence
        )

    def _search_financial_portals(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Real financial research using Yahoo Finance JSON APIs.

        A Wall Street analyst doesn't care about domain names — they care about
        the DATA.  This method fetches actual financial metrics (P/E, market
        cap, analyst targets, EPS growth, revenue growth, short interest, ROE)
        directly from Yahoo Finance's JSON APIs and puts that real data into
        every source's abstract.

        NO template URL generation. Every source either has real content
        (from a live API call) or is omitted entirely.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        # ── 1. Yahoo Finance search API — real JSON response ──────────────────
        yf_params = urllib.parse.urlencode(
            {
                "q": query,
                "quotesCount": min(target_limit, 12),
                "newsCount": min(target_limit, 12),
                "enableNavLinks": "true",
                "enableEnhancedTrivialQuery": "true",
            }
        )
        yf_payload = self._get_json(
            f"https://query1.finance.yahoo.com/v1/finance/search?{yf_params}"
        )

        # For each equity/ETF result, fetch REAL fundamental data
        fetched_symbols: set[str] = set()
        for item in (yf_payload.get("quotes") or [])[:target_limit]:
            symbol = str(item.get("symbol") or "").strip()
            name = str(item.get("longname") or item.get("shortname") or symbol)
            quote_type = str(item.get("quoteType") or "").upper()
            if not symbol or quote_type not in {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}:
                continue
            fetched_symbols.add(symbol)

            # Attempt to fetch real fundamental data — no fake placeholders
            abstract = self._fetch_yf_fundamentals_abstract(symbol, name)
            if not abstract:
                # API unavailable for this symbol — skip rather than fake it
                continue
            sources.append(
                ResearchSource(
                    provider="financial-portals",
                    title=f"{name} ({symbol}) — Live Fundamentals",
                    url=f"https://finance.yahoo.com/quote/{symbol}",
                    year=datetime.now(UTC).year,
                    abstract=abstract,
                    score=65.0,
                )
            )

        query_entity_terms = {
            token.lower() for token in re.findall(r"\b[A-Z][A-Za-z]{3,}\b", query)
        }
        query_entity_terms.update(
            token.lower() for token in _extract_ticker_candidates(query)
        )

        # Real news articles from Yahoo Finance (actual URLs, actual titles)
        for item in (yf_payload.get("news") or [])[: min(target_limit, 12)]:
            url = str(item.get("link") or "").strip()
            title = str(item.get("title") or "").strip()
            publisher = str(item.get("publisher") or "Yahoo Finance").strip()
            pub_time = item.get("providerPublishTime")
            if not url or not title or not url.startswith("http"):
                continue
            if not self._is_safe_public_url(url):
                continue
            host = urllib.parse.urlparse(url).netloc.lower()
            if host.endswith("yahoo.com"):
                continue
            title_text = f"{title} {publisher}".lower()
            mentions_entity = any(
                term in title_text or term in url.lower() for term in query_entity_terms
            )
            aligned_news = self._objective_alignment_score(title_text, query) >= 0.35
            if query_entity_terms and not (mentions_entity or aligned_news):
                continue
            year = datetime.now(UTC).year
            if pub_time:
                try:
                    year = datetime.fromtimestamp(int(pub_time), UTC).year
                except (ValueError, OSError):
                    pass
            sources.append(
                ResearchSource(
                    provider="financial-portals",
                    title=title,
                    url=url,
                    year=year,
                    abstract=f"{publisher}: {title}",
                    score=25.0,
                )
            )

        # ── 2. Explicit ticker extraction + company-name search ──────────────
        # Two strategies to resolve companies from the query:
        # (a) Short all-caps tickers via _extract_ticker_candidates
        # (b) Capitalized company-name tokens (e.g. NVIDIA, Apple, Microsoft)
        #     searched individually on YF v1 since the full query returns no quotes.
        tickers = _extract_ticker_candidates(query)
        candidate_names: list[str] = []
        _common_words = {
            "AND",
            "THE",
            "FOR",
            "WITH",
            "FROM",
            "THAT",
            "THIS",
            "STOCK",
            "STOCKS",
            "SHARE",
            "SHARES",
            "MARKET",
            "PRICE",
            "EARNINGS",
            "REVENUE",
            "GROWTH",
            "ANALYSIS",
            "REPORT",
            "QUARTER",
            "ANNUAL",
            "COMPANY",
            "CORP",
            "INC",
            "LLC",
            "LTD",
            "ABOUT",
            "WILL",
            "HAVE",
        }
        for token in query.split():
            t = token.strip(".,!?;:")
            # All-caps 2-10 chars that aren't blocked words (company names)
            if re.match(r"^[A-Z]{2,10}$", t) and t not in _common_words:
                candidate_names.append(t)
            # Title-case 5+ chars (Apple, Google, Microsoft, Amazon …)
            elif re.match(r"^[A-Z][a-z]{4,}$", t):
                candidate_names.append(t)

        for name_cand in candidate_names[:5]:
            if name_cand.upper() in fetched_symbols:
                continue
            yf_p = urllib.parse.urlencode(
                {"q": name_cand, "quotesCount": 3, "newsCount": 0}
            )
            yf_data = self._get_json(
                f"https://query1.finance.yahoo.com/v1/finance/search?{yf_p}"
            )
            for item in (yf_data.get("quotes") or [])[:2]:
                sym = str(item.get("symbol") or "").strip()
                nm = str(item.get("longname") or item.get("shortname") or sym)
                qt = str(item.get("quoteType") or "").upper()
                if not sym or qt not in {"EQUITY", "ETF", "MUTUALFUND"}:
                    continue
                if sym in fetched_symbols:
                    continue
                fetched_symbols.add(sym)
                abstract = self._fetch_yf_fundamentals_abstract(sym, nm)
                if not abstract:
                    continue
                sources.append(
                    ResearchSource(
                        provider="financial-portals",
                        title=f"{nm} ({sym}) — Live Fundamentals",
                        url=f"https://finance.yahoo.com/quote/{sym}",
                        year=datetime.now(UTC).year,
                        abstract=abstract,
                        score=65.0,
                    )
                )

        # Direct ticker lookup (short 1-5 char tickers explicitly in query)
        for ticker in tickers[:6]:
            ticker_upper = ticker.upper().strip()
            if ticker_upper in fetched_symbols:
                continue
            abstract = self._fetch_yf_fundamentals_abstract(ticker_upper, ticker_upper)
            if not abstract:
                continue
            fetched_symbols.add(ticker_upper)
            sources.append(
                ResearchSource(
                    provider="financial-portals",
                    title=f"{ticker_upper} — Yahoo Finance Fundamentals",
                    url=f"https://finance.yahoo.com/quote/{ticker_upper}",
                    year=datetime.now(UTC).year,
                    abstract=abstract,
                    score=60.0,
                )
            )

        self._record_provider_diagnostic(
            "financial-portals",
            "ok" if sources else "empty",
            f"returned {len(sources)} sources (real API data) for: {query[:80]}",
        )
        sources.sort(
            key=lambda source: (
                source.score,
                "Live Fundamentals" in source.title,
            ),
            reverse=True,
        )
        return sources[: target_limit * 3]

    def _search_sec_edgar(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Search SEC EDGAR for real company filings using browse-edgar Atom XML.

        The EFTS full-text search (efts.sec.gov) is blocked by Akamai CDN for
        programmatic access.  This method uses the officially-supported
        browse-edgar endpoint which returns Atom XML and is not CDN-protected:

        1. For tickers found in the query: resolve CIK via company_tickers.json
           then call browse-edgar for the exact company's filing history.
        2. For company-name searches: use browse-edgar's company search mode.
        3. For named tickers: call _fetch_sec_company_facts for full XBRL data
           (audited revenue, net income, EPS) — the primary-source numbers a
           sell-side analyst builds their model from.
        """
        import xml.etree.ElementTree as ET

        target_limit = max(1, int(limit or self.limit_per_provider))
        sources: list[ResearchSource] = []
        ATOM_NS = "http://www.w3.org/2005/Atom"
        sec_ua = os.environ.get(
            "SEC_USER_AGENT",
            "agentos-orchestrator/0.1 research-bot (public-research-use)",
        )

        def _fetch_browse_edgar_atom(url: str) -> "list[ResearchSource]":
            """Fetch browse-edgar Atom XML and parse into ResearchSource objects."""
            body = self._get_sec_text(
                url,
                accept="application/atom+xml,text/xml,*/*",
            )
            if not body:
                return []
            try:
                root = ET.fromstring(body)
            except ET.ParseError:
                return []

            atom_tag = f"{{{ATOM_NS}}}"
            # Company name from optional <company-info> element
            ci = root.find("company-info") or root.find(f"{atom_tag}company-info")
            company_name = ""
            if ci is not None:
                company_name = (
                    ci.findtext("conformed-name")
                    or ci.findtext(f"{atom_tag}conformed-name")
                    or ""
                )

            parsed: list[ResearchSource] = []
            for entry_el in list(root.iter(f"{atom_tag}entry"))[:12]:
                title = entry_el.findtext(f"{atom_tag}title") or ""
                link_el = entry_el.find(f"{atom_tag}link")
                href = link_el.get("href", "") if link_el is not None else ""
                updated = entry_el.findtext(f"{atom_tag}updated") or ""
                cat_el = entry_el.find(f"{atom_tag}category")
                form_type = (
                    cat_el.get("term", "Filing") if cat_el is not None else "Filing"
                )
                if not href or not href.startswith("http"):
                    continue
                year = int(updated[:4]) if len(updated) >= 4 else datetime.now(UTC).year
                label = company_name or form_type
                parsed.append(
                    ResearchSource(
                        provider="sec-edgar",
                        title=f"{label} — {form_type} ({updated[:10]})",
                        url=href,
                        year=year,
                        abstract=(
                            f"SEC {form_type} filing"
                            + (f" by {company_name}" if company_name else "")
                            + f". Filed: {updated[:10]}."
                            + (f" {title[:200]}" if title else "")
                        ),
                        score=52.0,
                    )
                )
            return parsed

        tickers = _extract_ticker_candidates(query)

        # ── 1. Per-ticker: direct CIK lookup → browse-edgar filing list + XBRL
        seen_ciks: set[str] = set()
        for ticker in tickers[:4]:
            ticker_up = ticker.upper().strip()

            # High-value XBRL data (audited P&L from actual 10-K filings)
            sec_source = self._fetch_sec_company_facts(ticker_up)
            if sec_source is not None:
                sources.append(sec_source)

            # Filing list via browse-edgar CIK lookup
            tickers_map = self._get_sec_tickers()
            entry = tickers_map.get(ticker_up)
            if entry:
                cik_int = entry.get("cik_str") or entry.get("cik")
                if cik_int:
                    cik_str = str(int(cik_int))
                    if cik_str not in seen_ciks:
                        seen_ciks.add(cik_str)
                        atom_url = (
                            f"https://www.sec.gov/cgi-bin/browse-edgar"
                            f"?action=getcompany&CIK={cik_str}"
                            f"&type=10-K,10-Q,8-K&dateb=&owner=include"
                            f"&count={target_limit}&output=atom"
                        )
                        sources.extend(_fetch_browse_edgar_atom(atom_url))

        # ── 2. Company-name search on browse-edgar (catches non-ticker queries)
        # Browse-edgar company search needs a focused company name, NOT a full
        # analytical query.  Extract the first recognizable company name token.
        if not tickers or len(sources) < target_limit:
            # Extract the best company-name term from the query
            _stop = {
                "AND",
                "THE",
                "FOR",
                "WITH",
                "FROM",
                "THAT",
                "THIS",
                "ABOUT",
                "WILL",
                "HAVE",
                "STOCK",
                "STOCKS",
                "SHARES",
                "MARKET",
                "PRICE",
                "EARNINGS",
                "REVENUE",
                "GROWTH",
                "ANALYSIS",
                "REPORT",
                "QUARTER",
                "ANNUAL",
                "QUARTERLY",
                "COMPANY",
                "CORP",
                "INC",
            }
            company_search_term = query[:60]
            for token in query.split():
                t = token.strip(".,!?;:")
                # Prefer long all-caps tokens (NVIDIA, AMAZON, MICROSOFT …)
                if re.match(r"^[A-Z]{3,}$", t) and t not in _stop:
                    company_search_term = t
                    break
                # Or title-case words of 5+ chars
                if re.match(r"^[A-Z][a-z]{4,}$", t):
                    company_search_term = t
                    break
            search_url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?company={urllib.parse.quote_plus(company_search_term)}"
                f"&CIK=&type=10-K&dateb=&owner=include"
                f"&count={target_limit}&search_text=&action=getcompany&output=atom"
            )
            sources.extend(_fetch_browse_edgar_atom(search_url))

        self._record_provider_diagnostic(
            "sec-edgar",
            "ok" if sources else "empty",
            f"returned {len(sources)} SEC sources for: {query[:80]}",
        )
        return sources

    # ──────────────────────────────────────────────────────────────────────────
    # WALL STREET RESEARCH PROVIDERS
    # Real data sources a sell-side analyst would use to build a research note.
    # Every method here fetches actual numbers from live APIs or public pages —
    # no templates, no placeholder URLs.
    # ──────────────────────────────────────────────────────────────────────────

    def _search_bing_results(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Bing web search — second engine alongside DuckDuckGo.

        Bing covers many financial/news pages that DuckDuckGo doesn't surface,
        especially WSJ, Bloomberg, Reuters, and MarketWatch articles.
        Uses HTML scraping of Bing's public search page (no API key needed).
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        seen_urls: set[str] = set()

        # Bing requires realistic browser headers to avoid bot-detection responses.
        bing_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.bing.com/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
        }

        for page in range(min(3, (target_limit + 9) // 10)):
            first_param = page * 10
            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "first": str(first_param) if first_param > 0 else "1",
                    "setlang": "en-US",
                    "cc": "US",
                    "mkt": "en-US",
                }
            )
            raw = self._get_text(
                f"https://www.bing.com/search?{params}",
                accept="text/html,application/xhtml+xml",
                max_bytes=200_000,
                timeout_seconds=10,
                extra_headers=bing_headers,
            )
            if not raw:
                break

            added_this_page = 0
            # Bing 2024+ results: multiple possible result structures.
            # Pattern 1: <h2><a href="...">title</a></h2> inside .b_algo
            # Pattern 2: data-href attributes in newer layouts
            patterns = [
                r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"&]+)"[^>]*>(.*?)</a>',
                r'<a[^>]+class="[^"]*tilk[^"]*"[^>]+href="(https?://[^"&]+)"[^>]*>(.*?)</a>',
                r'cite[^>]*>(https?://[^<]+)</cite>',
            ]
            # Use the richest pattern that gives results
            for pattern in patterns:
                matches = list(re.finditer(pattern, raw, flags=re.IGNORECASE | re.DOTALL))
                if matches:
                    for match in matches:
                        if len(match.groups()) == 2:
                            url = match.group(1).strip()
                            title = self._html_to_text(match.group(2)).strip()
                        else:
                            url = "https://" + match.group(1).strip().lstrip("https://")
                            title = self._label_from_url(url)
                        if not url or not title:
                            continue
                        if not self._is_safe_public_url(url):
                            continue
                        if "bing.com" in url or "microsoft.com" in url:
                            continue
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        # Extract snippet from surrounding HTML
                        match_end = match.end()
                        tail = raw[match_end: match_end + 3000]
                        snippet_m = re.search(
                            r'<p[^>]*(?:class="[^"]*(?:b_algoSlug|b_paractl|snippet)[^"]*")?[^>]*>(.*?)</p>',
                            tail,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        snippet = self._html_to_text(snippet_m.group(1)).strip() if snippet_m else ""
                        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
                        score = max(target_limit - len(sources), 1)
                        sources.append(
                            ResearchSource(
                                provider="bing-search",
                                title=title[:160],
                                url=url,
                                authors=[host] if host else [],
                                abstract=(snippet or f"Bing result: {title}")[:800],
                                score=float(score),
                            )
                        )
                        added_this_page += 1
                        if len(sources) >= target_limit:
                            break
                    if sources:  # found results with this pattern
                        break
            if len(sources) >= target_limit or added_this_page == 0:
                break

        self._record_provider_diagnostic(
            "bing-search",
            "ok" if sources else "empty",
            f"returned {len(sources)} results",
        )
        return sources

    def _search_google_news_rss(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Google News RSS — real-time financial news with actual article links.

        Google News RSS is public and unauthenticated. Returns genuine news
        articles from WSJ, Reuters, Bloomberg, CNBC, FT, MarketWatch, etc.
        — the same sources a Bloomberg terminal's news feed would show.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        # Encode query for Google News RSS
        params = urllib.parse.urlencode(
            {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        rss_url = f"https://news.google.com/rss/search?{params}"
        # Google News blocks plain Python user-agents; use a realistic browser UA.
        raw = self._get_text(
            rss_url,
            accept="application/rss+xml,text/xml,application/xml,*/*;q=0.8",
            max_bytes=200_000,
            timeout_seconds=10,
            extra_headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://news.google.com/",
                "Cache-Control": "no-cache",
            },
        )
        if not raw or len(raw) < 100:
            self._record_provider_diagnostic(
                "google-news-rss", "empty", "no RSS response"
            )
            return []

        # Parse RSS <item> elements
        seen: set[str] = set()
        for item_match in re.finditer(
            r"<item>(.*?)</item>", raw, re.DOTALL | re.IGNORECASE
        ):
            item = item_match.group(1)
            title_m = re.search(
                r"<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>",
                item,
                re.DOTALL,
            )
            link_m = re.search(
                r"<link>(.*?)</link>|<guid[^>]*>(https?://[^<]+)</guid>",
                item,
                re.DOTALL,
            )
            pub_m = re.search(r"<pubDate>(.*?)</pubDate>", item)
            source_m = re.search(r"<source[^>]*>(.*?)</source>", item, re.DOTALL)
            title = (
                (title_m.group(1) or title_m.group(2) or "").strip() if title_m else ""
            )
            url = (link_m.group(1) or link_m.group(2) or "").strip() if link_m else ""
            # Google News wraps links in its redirect; try to decode
            if "news.google.com" in url:
                # The real URL is after a redirect; use as-is (enrichment will follow)
                pass
            publisher = (
                self._html_to_text(source_m.group(1)).strip()
                if source_m
                else "Google News"
            )
            pub_date = pub_m.group(1).strip() if pub_m else ""
            year = datetime.now(UTC).year
            if pub_date:
                yr_m = re.search(r"\b(20\d\d)\b", pub_date)
                if yr_m:
                    year = int(yr_m.group(1))
            if not title or not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            sources.append(
                ResearchSource(
                    provider="google-news-rss",
                    title=title[:160],
                    url=url,
                    year=year,
                    authors=[publisher] if publisher else [],
                    abstract=f"{publisher}: {title}",
                    score=30.0,
                )
            )
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "google-news-rss",
            "ok" if sources else "empty",
            f"returned {len(sources)} news articles",
        )
        return sources

    def _search_macrotrends(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Macrotrends.net — historical financial metrics without API key.

        Macrotrends is the go-to source for 10–20-year historical charts of
        P/E ratio, revenue, net income, EPS, gross margin, EBITDA, dividend
        yield, and 50+ other metrics. A real analyst ALWAYS pulls Macrotrends
        to understand long-run valuation context and trend direction.

        Extracts company name tokens from the query, maps to Macrotrends
        ticker/slug URLs, and fetches the actual page content.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        # Resolve company name → ticker via yfinance if available
        candidate_tickers: list[tuple[str, str]] = []  # (ticker, name)
        try:
            import yfinance as yf  # type: ignore[import-not-found]

            _candidates = _extract_ticker_candidates(query)
            _stop = {
                "AND",
                "THE",
                "FOR",
                "WITH",
                "FROM",
                "THAT",
                "THIS",
                "STOCK",
                "EARNINGS",
                "REVENUE",
                "GROWTH",
                "ANALYSIS",
            }
            for token in query.split():
                t = token.strip(".,!?;:")
                if re.match(r"^[A-Z]{2,10}$", t) and t not in _stop:
                    if t not in _candidates:
                        _candidates.append(t)
                elif re.match(r"^[A-Z][a-z]{4,}$", t):
                    # Title-case: search YF to get ticker
                    yf_p = urllib.parse.urlencode(
                        {"q": t, "quotesCount": 1, "newsCount": 0}
                    )
                    yf_d = self._get_json(
                        f"https://query1.finance.yahoo.com/v1/finance/search?{yf_p}"
                    )
                    for item in (yf_d.get("quotes") or [])[:1]:
                        sym = str(item.get("symbol") or "").strip()
                        if sym and str(item.get("quoteType") or "").upper() == "EQUITY":
                            _candidates.append(sym)
            for ticker in _candidates[:4]:
                try:
                    info = yf.Ticker(ticker).info or {}
                    name = str(info.get("longName") or info.get("shortName") or ticker)
                    candidate_tickers.append((ticker, name))
                except Exception:
                    candidate_tickers.append((ticker, ticker))
        except ImportError:
            _tickers = _extract_ticker_candidates(query)
            candidate_tickers = [(t, t) for t in _tickers[:4]]

        # Macrotrends metric pages — these have actual historical series
        metrics = [
            ("revenue", "Revenue"),
            ("net-income", "Net Income"),
            ("eps-earnings-per-share-diluted", "Diluted EPS"),
            ("pe-ratio", "P/E Ratio"),
            ("price-to-book-ratio", "P/B Ratio"),
            ("return-on-equity", "ROE"),
            ("free-cash-flow", "Free Cash Flow"),
            ("gross-profit-margin", "Gross Margin"),
        ]

        for ticker, name in candidate_tickers[:3]:
            ticker_lc = ticker.lower()
            name_slug = re.sub(r"[^a-z0-9]+", "-", name.lower())[:30].strip("-")
            for metric_slug, metric_label in metrics[:4]:  # top 4 metrics per company
                url = f"https://www.macrotrends.net/stocks/charts/{ticker_lc}/{name_slug}/{metric_slug}"
                raw = self._get_text(
                    url, accept="text/html,*/*", max_bytes=80_000, timeout_seconds=8
                )
                if not raw:
                    continue
                # Extract JSON data embedded in the page
                # Macrotrends embeds data as: var originalData = [...];
                data_m = re.search(
                    r"var\s+originalData\s*=\s*(\[.*?\]);", raw, re.DOTALL
                )
                if data_m:
                    try:
                        data_rows = json.loads(data_m.group(1))
                        # Each row: {"date": "2024-01-01", "v1": 123456789}
                        rows = [
                            (r.get("date", "")[:7], r.get("v1") or r.get("v2"))
                            for r in data_rows
                            if r.get("date")
                        ]
                        rows = [(d, v) for d, v in rows if v is not None]
                        if rows:
                            recent = rows[-8:]  # last 8 quarters/years
                            series_str = "; ".join(
                                f"{d}: ${v / 1e9:.2f}B"
                                if abs(float(v)) >= 1e9
                                else f"{d}: {float(v):.2f}"
                                if metric_slug.endswith("ratio")
                                or metric_slug.startswith("pe")
                                or "margin" in metric_slug
                                else f"{d}: ${v / 1e6:.0f}M"
                                for d, v in recent
                            )
                            abstract = f"{name} ({ticker}) {metric_label} — Historical Data: {series_str}"
                            sources.append(
                                ResearchSource(
                                    provider="macrotrends",
                                    title=f"{name} ({ticker}) — {metric_label} Historical",
                                    url=url,
                                    year=datetime.now(UTC).year,
                                    abstract=abstract[:2000],
                                    score=55.0,
                                )
                            )
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                elif raw:
                    # No embedded data — still useful as a page to enrich
                    text = self._html_to_text(raw)[:800]
                    if text and len(text) > 100 and name.lower() in text.lower():
                        sources.append(
                            ResearchSource(
                                provider="macrotrends",
                                title=f"{name} ({ticker}) — {metric_label}",
                                url=url,
                                year=datetime.now(UTC).year,
                                abstract=text[:800],
                                score=40.0,
                            )
                        )
                if len(sources) >= target_limit:
                    break
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "macrotrends",
            "ok" if sources else "empty",
            f"returned {len(sources)} historical metric sources",
        )
        return sources

    def _search_stockanalysis(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """StockAnalysis.com — free earnings, revenue, balance sheet data.

        StockAnalysis provides clean tables of quarterly/annual financials,
        earnings estimates, institutional ownership, and more. It renders
        as static HTML for many sections, making it scrapable without a
        JavaScript engine. A key data cross-check tool for any analyst.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        seen_urls: set[str] = set()

        # Extract tickers
        tickers: list[str] = []
        _stop = {
            "AND",
            "THE",
            "FOR",
            "WITH",
            "STOCK",
            "EARNINGS",
            "REVENUE",
            "GROWTH",
            "ANALYSIS",
            "MARKET",
            "SHARES",
        }
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,10}$", t) and t not in _stop:
                tickers.append(t)
            elif re.match(r"^[A-Z][a-z]{4,}$", t):
                # Title-case name: look up ticker
                yf_p = urllib.parse.urlencode(
                    {"q": t, "quotesCount": 1, "newsCount": 0}
                )
                yf_d = self._get_json(
                    f"https://query1.finance.yahoo.com/v1/finance/search?{yf_p}"
                )
                for item in (yf_d.get("quotes") or [])[:1]:
                    sym = str(item.get("symbol") or "").strip()
                    if sym:
                        tickers.append(sym)

        tickers = list(dict.fromkeys(tickers))[:4]

        sections = [
            ("financials", "Annual Financials"),
            ("financials/quarterly", "Quarterly Financials"),
            ("forecast", "Earnings Estimates"),
        ]

        for ticker in tickers[:3]:
            ticker_lc = ticker.lower()
            for section_path, section_label in sections[:2]:
                url = f"https://stockanalysis.com/stocks/{ticker_lc}/{section_path}/"
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                raw = self._get_text(
                    url, accept="text/html,*/*", max_bytes=100_000, timeout_seconds=8
                )
                if not raw:
                    continue
                text = self._html_to_text(raw)
                if len(text) < 200:
                    continue
                # Extract the financial table portion
                # Look for revenue/earnings patterns in extracted text
                numbers = re.findall(r"\b\d{1,3}(?:\.\d+)?[BMK%]\b", text)
                if not numbers:
                    continue
                # Build a compact summary of financial data found
                first_2000 = re.sub(r"\s+", " ", text)[:2000]
                sources.append(
                    ResearchSource(
                        provider="stockanalysis",
                        title=f"{ticker.upper()} — {section_label} (StockAnalysis)",
                        url=url,
                        year=datetime.now(UTC).year,
                        abstract=first_2000,
                        score=58.0,
                    )
                )
                if len(sources) >= target_limit:
                    break
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "stockanalysis",
            "ok" if sources else "empty",
            f"returned {len(sources)} financial table sources",
        )
        return sources

    def _search_insider_transactions(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """SEC Form 4 insider transactions — real smart-money signals.

        When executives and directors buy or sell their own stock, that's
        material non-public-activity-adjacent signal. SEC Form 4s are filed
        within 2 business days of every insider transaction and are public
        record. This is THE primary source for insider-trading analysis on
        Wall Street.

        Uses OpenInsider.com (aggregates SEC Form 4 filings, freely accessible)
        and the SEC EDGAR full-text browse endpoint for Form 4 filings.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        # Extract tickers from query
        tickers: list[str] = []
        _stop = {
            "AND",
            "THE",
            "FOR",
            "INSIDER",
            "TRANSACTION",
            "STOCK",
            "TRADING",
            "BUYING",
            "SELLING",
        }
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,6}$", t) and t not in _stop:
                tickers.append(t)
        tickers = _extract_ticker_candidates(query) + tickers
        tickers = list(dict.fromkeys(tickers))[:4]

        for ticker in tickers[:3]:
            # OpenInsider — aggregated Form 4 data in human-readable table
            oi_url = f"https://openinsider.com/search?q={urllib.parse.quote(ticker)}"
            raw = self._get_text(
                oi_url, accept="text/html,*/*", max_bytes=80_000, timeout_seconds=8
            )
            if raw:
                text = self._html_to_text(raw)
                # Look for buy/sell transactions
                buy_m = re.findall(
                    r"(?:purchase|buy|bought)[^.]*\$[\d,]+", text.lower()
                )
                sell_m = re.findall(r"(?:sale|sell|sold)[^.]*\$[\d,]+", text.lower())
                if text and len(text) > 200:
                    sources.append(
                        ResearchSource(
                            provider="insider-transactions",
                            title=f"{ticker.upper()} — SEC Form 4 Insider Transactions",
                            url=oi_url,
                            year=datetime.now(UTC).year,
                            abstract=re.sub(r"\s+", " ", text)[:1500],
                            score=62.0,
                        )
                    )

            # Also pull directly from SEC EDGAR Form 4 browse
            tickers_map = self._get_sec_tickers()
            entry = tickers_map.get(ticker.upper())
            if entry:
                cik_int = entry.get("cik_str") or entry.get("cik")
                if cik_int:
                    cik_str = str(int(cik_int))
                    form4_url = (
                        f"https://www.sec.gov/cgi-bin/browse-edgar"
                        f"?action=getcompany&CIK={cik_str}"
                        f"&type=4&dateb=&owner=include&count={target_limit}"
                        f"&search_text=&output=atom"
                    )
                    form4_sources = self._fetch_browse_edgar_atom_generic(
                        form4_url, "4", entry.get("title", ticker)
                    )
                    sources.extend(form4_sources[:3])

            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "insider-transactions",
            "ok" if sources else "empty",
            f"returned {len(sources)} insider transaction sources",
        )
        return sources

    def _fetch_browse_edgar_atom_generic(
        self,
        url: str,
        form_type: str,
        company_name: str,
    ) -> list[ResearchSource]:
        """Reusable browse-edgar Atom XML fetcher for any form type."""
        import xml.etree.ElementTree as ET

        ATOM_NS = "http://www.w3.org/2005/Atom"
        body = self._get_sec_text(
            url,
            accept="application/atom+xml,text/xml,*/*",
        )
        if not body:
            return []
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return []
        sources: list[ResearchSource] = []
        for entry_el in list(root.iter(f"{{{ATOM_NS}}}entry"))[:8]:
            link_el = entry_el.find(f"{{{ATOM_NS}}}link")
            href = link_el.get("href", "") if link_el is not None else ""
            updated = entry_el.findtext(f"{{{ATOM_NS}}}updated") or ""
            if not href or not href.startswith("http"):
                continue
            year = int(updated[:4]) if len(updated) >= 4 else datetime.now(UTC).year
            sources.append(
                ResearchSource(
                    provider="sec-edgar",
                    title=f"{company_name} — Form {form_type} ({updated[:10]})",
                    url=href,
                    year=year,
                    abstract=f"SEC Form {form_type} filing by {company_name}. Filed: {updated[:10]}.",
                    score=48.0,
                )
            )
        return sources

    def _search_short_interest(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Short interest data from FINRA and public financial sites.

        Short interest is one of the most important signals for identifying:
        1. Potential short squeezes (high short interest + rising price)
        2. Market consensus that a stock is overvalued
        3. Hidden risk in positions

        FINRA publishes bi-monthly short interest data for all exchange-listed
        securities and it's freely accessible. We also check finviz and
        Yahoo Finance (already have via yfinance short float).
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        tickers: list[str] = []
        _stop = {"AND", "THE", "FOR", "SHORT", "INTEREST", "SQUEEZE", "FLOAT"}
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,6}$", t) and t not in _stop:
                tickers.append(t)
        tickers = _extract_ticker_candidates(query) + tickers
        tickers = list(dict.fromkeys(tickers))[:4]

        for ticker in tickers[:3]:
            # Finviz has real-time short float and short ratio in their screener
            finviz_url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
            raw = self._get_text(
                finviz_url, accept="text/html,*/*", max_bytes=80_000, timeout_seconds=8
            )
            if raw:
                text = self._html_to_text(raw)
                # Look for key Finviz table values
                short_float_m = re.search(
                    r"Short Float[^\d]*([\d.]+%)", raw, re.IGNORECASE
                )
                short_ratio_m = re.search(
                    r"Short Ratio[^\d]*([\d.]+)", raw, re.IGNORECASE
                )
                short_int_m = re.search(
                    r"Short Interest[^\d]*([\d,.]+[MK]?)", raw, re.IGNORECASE
                )
                shares_outstanding_m = re.search(
                    r"Shs Outstand[^\d]*([\d.]+[MBK]?)", raw, re.IGNORECASE
                )
                float_m = re.search(
                    r"(?:^|\s)Float[^\d]*([\d.]+[MBK]?)", raw, re.IGNORECASE
                )
                inst_own_m = re.search(r"Inst Own[^\d]*([\d.]+%)", raw, re.IGNORECASE)
                insider_own_m = re.search(
                    r"Insider Own[^\d]*([\d.]+%)", raw, re.IGNORECASE
                )
                perf_ytd_m = re.search(r"Perf YTD[^\s]*([-+\d.]+%)", raw, re.IGNORECASE)

                parts: list[str] = [f"{ticker.upper()} (Finviz)"]
                if short_float_m:
                    parts.append(f"Short Float: {short_float_m.group(1)}")
                if short_ratio_m:
                    parts.append(f"Short Ratio: {short_ratio_m.group(1)} days")
                if short_int_m:
                    parts.append(f"Short Interest: {short_int_m.group(1)}")
                if float_m:
                    parts.append(f"Float: {float_m.group(1)}")
                if inst_own_m:
                    parts.append(f"Inst Own: {inst_own_m.group(1)}")
                if insider_own_m:
                    parts.append(f"Insider Own: {insider_own_m.group(1)}")
                if perf_ytd_m:
                    parts.append(f"Perf YTD: {perf_ytd_m.group(1)}")

                if len(parts) > 2:
                    sources.append(
                        ResearchSource(
                            provider="short-interest",
                            title=f"{ticker.upper()} — Short Interest & Ownership (Finviz)",
                            url=finviz_url,
                            year=datetime.now(UTC).year,
                            abstract=" | ".join(parts),
                            score=60.0,
                        )
                    )
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "short-interest",
            "ok" if sources else "empty",
            f"returned {len(sources)} short interest sources",
        )
        return sources

    def _search_earnings_data(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Earnings estimates and calendar from Yahoo Finance earnings API.

        Earnings surprises (beat vs miss) and forward guidance revisions are
        the single most important short-term catalyst for stock prices. This
        method fetches the earnings calendar and analyst EPS estimates — the
        same data a sell-side analyst builds their quarterly model around.
        Uses yfinance for earnings history and the YF earnings calendar API.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        tickers: list[str] = []
        _stop = {
            "AND",
            "THE",
            "FOR",
            "EARNINGS",
            "CALENDAR",
            "ESTIMATE",
            "EPS",
            "GUIDANCE",
            "BEAT",
            "MISS",
        }
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,10}$", t) and t not in _stop:
                tickers.append(t)
        tickers = _extract_ticker_candidates(query) + tickers
        tickers = list(dict.fromkeys(tickers))[:4]

        try:
            import yfinance as yf  # type: ignore[import-not-found]

            for ticker in tickers[:4]:
                try:
                    tk = yf.Ticker(ticker.upper())
                    # Earnings history
                    history = tk.earnings_history
                    if history is not None and not history.empty:
                        rows = history.tail(8)
                        lines: list[str] = []
                        for idx, row in rows.iterrows():
                            eps_est = row.get("epsEstimate")
                            eps_act = row.get("epsActual")
                            surprise = row.get("epsDifference")
                            surprise_pct = row.get("surprisePercent")
                            if eps_act is not None:
                                period = str(idx)[:10] if idx else "?"
                                line = f"{period}: EPS ${eps_act:.2f}"
                                if eps_est is not None:
                                    line += f" (est ${eps_est:.2f}"
                                if surprise_pct is not None:
                                    line += f", {surprise_pct:+.1f}% surprise)"
                                elif eps_est is not None:
                                    line += ")"
                                lines.append(line)
                        if lines:
                            sources.append(
                                ResearchSource(
                                    provider="earnings-data",
                                    title=f"{ticker.upper()} — EPS History & Surprise Track",
                                    url=f"https://finance.yahoo.com/quote/{ticker.upper()}/earnings",
                                    year=datetime.now(UTC).year,
                                    abstract=f"{ticker.upper()} Earnings History: "
                                    + "; ".join(lines),
                                    score=68.0,
                                )
                            )
                    # Next earnings date estimate
                    cal = tk.calendar
                    if cal is not None and not cal.empty:
                        next_date = cal.get("Earnings Date")
                        if next_date is not None:
                            dates = (
                                list(next_date)
                                if hasattr(next_date, "__iter__")
                                else [next_date]
                            )
                            date_str = ", ".join(str(d)[:10] for d in dates[:2])
                            eps_est_low = cal.get("EPS Estimate")
                            rev_est = cal.get("Revenue Estimate")
                            cal_parts = [f"{ticker.upper()} Next Earnings: {date_str}"]
                            if eps_est_low is not None:
                                try:
                                    cal_parts.append(
                                        f"EPS Est: ${float(eps_est_low.iloc[0]):.2f}"
                                    )
                                except Exception:
                                    pass
                            if rev_est is not None:
                                try:
                                    rv = float(rev_est.iloc[0])
                                    cal_parts.append(f"Rev Est: ${rv / 1e9:.2f}B")
                                except Exception:
                                    pass
                            sources.append(
                                ResearchSource(
                                    provider="earnings-data",
                                    title=f"{ticker.upper()} — Next Earnings Date & Estimates",
                                    url=f"https://finance.yahoo.com/quote/{ticker.upper()}/earnings",
                                    year=datetime.now(UTC).year,
                                    abstract=" | ".join(cal_parts),
                                    score=65.0,
                                )
                            )
                except Exception:
                    pass
                if len(sources) >= target_limit:
                    break
        except ImportError:
            pass

        self._record_provider_diagnostic(
            "earnings-data",
            "ok" if sources else "empty",
            f"returned {len(sources)} earnings sources",
        )
        return sources

    def _search_fed_macro_data(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Federal Reserve (FRED) and BEA/BLS macro data — public APIs.

        Macro context is essential for equity valuation. Interest rates,
        inflation, GDP growth, unemployment, and yield curve shape all affect
        DCF discount rates and earnings multiples. This method fetches key
        macro indicators from FRED (Federal Reserve Economic Data) which
        provides a public JSON API for hundreds of economic series.

        A real macro/equity analyst checks these before forming a view on
        valuation multiples and risk premium.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        lower_q = query.lower()

        # FRED series IDs for macro indicators a WS analyst would check
        series_map = {
            "DFF": ("Fed Funds Rate (Effective)", "interest rate macro"),
            "GS10": ("10-Year Treasury Yield", "interest rate yield curve"),
            "GS2": ("2-Year Treasury Yield", "yield curve short rate"),
            "T10Y2Y": (
                "10Y-2Y Yield Spread (Inversion Signal)",
                "yield curve inversion",
            ),
            "CPIAUCSL": ("CPI Inflation (All Items)", "inflation price level"),
            "PCE": ("Personal Consumption Expenditures", "consumer spending gdp"),
            "GDPC1": ("Real GDP (Annualized)", "economic growth gdp"),
            "UNRATE": ("Unemployment Rate", "labor market jobs"),
            "VIXCLS": ("CBOE Volatility Index (VIX)", "market volatility fear"),
            "SP500": ("S&P 500 Index Level", "stock market equity"),
            "DEXUSEU": ("USD/EUR Exchange Rate", "dollar currency forex"),
            "DCOILWTICO": ("WTI Crude Oil Price", "oil energy commodity"),
        }

        # Determine which series are relevant to this query
        relevant_ids: list[str] = []
        for series_id, (label, keywords) in series_map.items():
            for kw in keywords.split():
                if kw in lower_q:
                    relevant_ids.append(series_id)
                    break

        # If no specific match, include core macro indicators for any WS query
        if not relevant_ids and self._looks_like_market_query(query):
            relevant_ids = ["DFF", "GS10", "T10Y2Y", "CPIAUCSL", "GDPC1", "VIXCLS"]

        if not relevant_ids:
            self._record_provider_diagnostic(
                "fed-macro", "skipped", "query not macro-relevant"
            )
            return []

        # FRED API — returns JSON, completely free and public
        fred_key = (
            os.environ.get("FRED_API_KEY") or ""
        )  # optional key for higher rate limits
        for series_id in relevant_ids[: min(target_limit, 6)]:
            label, _ = series_map[series_id]
            params = {
                "series_id": series_id,
                "file_type": "json",
                "limit": "8",
                "sort_order": "desc",
            }
            if fred_key:
                params["api_key"] = fred_key
            fred_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            # Use the public CSV endpoint (no API key needed)
            csv_raw = self._get_text(
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&vintage_date=",
                accept="text/csv,text/plain,*/*",
                max_bytes=20_000,
                timeout_seconds=6,
            )
            if not csv_raw:
                continue
            lines = [
                line.strip()
                for line in csv_raw.splitlines()
                if line.strip() and not line.startswith("DATE")
            ]
            # Last 8 data points
            recent_lines = [line for line in lines if "." in line][-8:]
            if not recent_lines:
                continue
            data_points: list[str] = []
            for line in recent_lines:
                parts = line.split(",")
                if (
                    len(parts) >= 2
                    and parts[1].replace(".", "").replace("-", "").isdigit()
                ):
                    data_points.append(f"{parts[0]}: {parts[1]}")
            if not data_points:
                continue
            abstract = f"{label}: " + "; ".join(data_points[-4:])
            sources.append(
                ResearchSource(
                    provider="fed-macro",
                    title=f"FRED: {label} ({series_id})",
                    url=f"https://fred.stlouisfed.org/series/{series_id}",
                    year=datetime.now(UTC).year,
                    abstract=abstract,
                    score=50.0,
                )
            )
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "fed-macro",
            "ok" if sources else "empty",
            f"returned {len(sources)} macro indicator sources",
        )
        return sources

    def _search_seeking_alpha_news(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Seeking Alpha public earnings articles and analysis.

        Seeking Alpha is the largest independent equity research platform.
        Their public pages (no paywall for news/earnings articles) contain:
        - Earnings call transcripts summaries
        - Analyst ratings changes
        - Company news articles
        These are the kind of qualitative research inputs a WS analyst reads
        to supplement the quantitative data.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        tickers: list[str] = []
        _stop = {"AND", "THE", "FOR", "ANALYSIS", "STOCK", "MARKET"}
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,6}$", t) and t not in _stop:
                tickers.append(t)
        tickers = _extract_ticker_candidates(query) + tickers
        tickers = list(dict.fromkeys(tickers))[:3]

        for ticker in tickers[:3]:
            # Seeking Alpha news page (public)
            sa_url = f"https://seekingalpha.com/symbol/{ticker.upper()}/news"
            raw = self._get_text(
                sa_url, accept="text/html,*/*", max_bytes=80_000, timeout_seconds=8
            )
            if raw:
                # Extract article titles and links
                article_links = re.findall(
                    r'href="(/article/[^"]+)"[^>]*>([^<]{20,200})', raw, re.IGNORECASE
                )
                for path, title_raw in article_links[:target_limit]:
                    title = self._html_to_text(title_raw).strip()
                    if not title:
                        continue
                    full_url = f"https://seekingalpha.com{path}"
                    sources.append(
                        ResearchSource(
                            provider="seeking-alpha",
                            title=title[:160],
                            url=full_url,
                            year=datetime.now(UTC).year,
                            abstract=f"Seeking Alpha analysis: {title}",
                            score=35.0,
                        )
                    )
                    if len(sources) >= target_limit:
                        break
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "seeking-alpha",
            "ok" if sources else "empty",
            f"returned {len(sources)} SA articles",
        )
        return sources

    def _search_reddit_finance(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Search Reddit finance communities for stock discussion and sentiment.

        Reddit's r/investing, r/stocks, r/wallstreetbets, and r/SecurityAnalysis
        provide real-time retail sentiment, DD (due diligence) posts, and often
        surface information before it reaches mainstream media.

        Uses Reddit's free JSON API — no authentication required for read-only.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        try:
            encoded = urllib.parse.quote_plus(query[:120])
            # Search across the 4 most relevant finance subreddits.
            subreddits = "investing+stocks+wallstreetbets+SecurityAnalysis"
            url = (
                f"https://www.reddit.com/r/{subreddits}/search.json"
                f"?q={encoded}&sort=top&t=month&limit={min(target_limit * 3, 25)}"
                f"&restrict_sr=1"
            )
            raw = self._get_text(
                url,
                accept="application/json",
                max_bytes=120_000,
                timeout_seconds=8,
                extra_headers={"Accept": "application/json"},
            )
            if not raw:
                raise ValueError("empty response")
            data = json.loads(raw)
            posts = (data.get("data") or {}).get("children") or []
            for post in posts:
                post_data = post.get("data") or {}
                title = (post_data.get("title") or "").strip()
                selftext = (post_data.get("selftext") or "").strip()[:500]
                permalink = post_data.get("permalink") or ""
                score_val = int(post_data.get("score") or 0)
                subreddit = post_data.get("subreddit") or "reddit"
                num_comments = int(post_data.get("num_comments") or 0)
                if not title or not permalink:
                    continue
                full_url = f"https://www.reddit.com{permalink}"
                abstract = selftext if selftext else f"Reddit discussion: {title}"
                # Higher upvote/comment count = more market signal.
                source_score = min(
                    45.0, 15.0 + (score_val / 500.0) + (num_comments / 20.0)
                )
                sources.append(
                    ResearchSource(
                        provider="reddit-finance",
                        title=title[:180],
                        url=full_url,
                        year=datetime.now(UTC).year,
                        authors=[f"r/{subreddit}"],
                        abstract=abstract,
                        score=source_score,
                        quality_flags=["social-media-signal"],
                    )
                )
                if len(sources) >= target_limit:
                    break
        except Exception as exc:
            self._record_provider_diagnostic("reddit-finance", "error", str(exc)[:120])
        self._record_provider_diagnostic(
            "reddit-finance",
            "ok" if sources else "empty",
            f"returned {len(sources)} Reddit posts",
        )
        return sources

    def _fetch_urls_async_httpx(
        self,
        urls: list[str],
        max_bytes: int = 60_000,
        timeout_seconds: float = 10.0,
    ) -> dict[str, str]:
        """Fetch multiple URLs in parallel using httpx async — dramatically faster
        than sequential requests for bulk URL fetching (e.g., chain expansion).

        Returns a dict mapping url → plain text content.
        Used by the chain expansion mechanism to process 100+ URLs at once.
        """
        import asyncio

        results: dict[str, str] = {}
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError:
            # Fall back to threaded requests if httpx is unavailable.
            def _fetch_one(url: str) -> tuple[str, str]:
                text = self._get_text(
                    url, max_bytes=max_bytes, timeout_seconds=int(timeout_seconds)
                )
                return url, text

            with ThreadPoolExecutor(max_workers=min(20, len(urls))) as pool:
                for url, text in pool.map(_fetch_one, urls):
                    if text:
                        results[url] = text
            return results

        async def _fetch_all(url_list: list[str]) -> None:
            ua = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
            limits = httpx.Limits(max_connections=30, max_keepalive_connections=10)
            async with httpx.AsyncClient(
                headers=headers,
                timeout=timeout_seconds,
                limits=limits,
                follow_redirects=True,
                verify=False,  # noqa: S501 — bulk fetching, no cert pinning needed
            ) as client:
                tasks = [client.get(u) for u in url_list]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for url, resp in zip(url_list, responses):
                    if isinstance(resp, Exception):
                        continue
                    try:
                        ct = resp.headers.get("content-type", "").lower()
                        if ct and not any(
                            m in ct for m in ("text/", "html", "json", "xml")
                        ):
                            continue
                        text = resp.text[:max_bytes]
                        if text:
                            results[url] = text
                    except Exception:
                        pass

        try:
            asyncio.run(_fetch_all(urls))
        except Exception:
            pass
        return results

    def _search_web_results(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        page_stride = 20
        max_pages = max(1, min(8, (target_limit + page_stride - 1) // page_stride + 1))
        seen_urls: set[str] = set()
        seen_pages: set[str] = set()
        global_rank = 0
        preview_fetch_budget = (
            12 if self._looks_like_current_evidence_query(query) else 6
        )
        preview_fetches = 0

        # DuckDuckGo HTML requires realistic browser headers to return results.
        ddg_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://html.duckduckgo.com/",
        }

        for page_index in range(max_pages):
            page_start = page_index * page_stride
            params: dict[str, str] = {"q": query}
            if page_start > 0:
                params["s"] = str(page_start)
                params["dc"] = str(page_start)
            search_url = (
                f"https://html.duckduckgo.com/html/?{urllib.parse.urlencode(params)}"
            )
            raw_html = self._get_text(
                search_url,
                accept="text/html,application/xhtml+xml",
                max_bytes=200_000,
                timeout_seconds=10,
                extra_headers=ddg_headers,
            )
            if not raw_html:
                break

            page_fingerprint = self._normalize_title(raw_html[:2400])
            if page_fingerprint in seen_pages:
                break
            seen_pages.add(page_fingerprint)

            added_this_page = 0
            # DuckDuckGo HTML: multiple possible result patterns (class names change)
            ddg_patterns = [
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                r'<a[^>]+class="[^"]*result-link[^"]*"[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
                r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            ]
            url_title_pairs: list[tuple[str, str, int]] = []
            added_this_page = 0
            for ddg_pat in ddg_patterns:
                for match in re.finditer(ddg_pat, raw_html, flags=re.IGNORECASE | re.DOTALL):
                    raw_url = self._normalize_web_result_url(match.group(1))
                    title = self._html_to_text(match.group(2)) or self._label_from_url(raw_url)
                    if raw_url and title:
                        url_title_pairs.append((raw_url, title, match.end()))
                if url_title_pairs:
                    break

            for raw_url, title, match_end in url_title_pairs:
                if not self._is_safe_public_url(raw_url):
                    continue
                if raw_url in seen_urls:
                    continue
                tail = raw_html[match_end: match_end + 1500]
                snippet_match = re.search(
                    r'class="[^"]*(?:result__snippet|result-snippet|snippet)[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
                    tail,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                snippet = (
                    self._html_to_text(snippet_match.group(1))
                    if snippet_match is not None
                    else ""
                )
                quality_flags: list[str] = []
                if not snippet.strip() and preview_fetches < preview_fetch_budget:
                    preview_fetches += 1
                    preview = self._get_text(
                        raw_url,
                        accept="text/html,application/xhtml+xml,*/*",
                        max_bytes=30_000,
                        timeout_seconds=4,
                    )
                    preview_text = self._html_to_text(preview)
                    tickers = _extract_ticker_candidates(f"{title} {preview_text}")
                    if tickers:
                        snippet = f"Ticker candidates mentioned: {', '.join(tickers)}."
                    elif len(preview_text) >= 80:
                        snippet = re.sub(r"\s+", " ", preview_text)[:320]
                    else:
                        quality_flags.append("snippet-unavailable")
                elif not snippet.strip():
                    quality_flags.append("snippet-unavailable")

                seen_urls.add(raw_url)
                added_this_page += 1
                global_rank += 1
                host = urllib.parse.urlparse(raw_url).netloc.lower().lstrip("www.")
                score = max(target_limit - global_rank + 1, 0)
                sources.append(
                    ResearchSource(
                        provider="web-search",
                        title=title[:160],
                        url=raw_url,
                        authors=[host] if host else [],
                        abstract=(
                            snippet or "Generic web result. Snippet unavailable."
                        )[:1200],
                        citation_count=score,
                        score=float(score),
                        quality_flags=quality_flags,
                    )
                )
                if len(sources) >= target_limit:
                    break
            if len(sources) >= target_limit or added_this_page == 0:
                break

        self._record_provider_diagnostic(
            "web-search",
            "ok" if sources else "empty",
            f"returned {len(sources)} results",
        )
        return sources

    # ------------------------------------------------------------------
    # Provider routing
    # ------------------------------------------------------------------

    def _search_query_across_providers(
        self,
        search_query: str,
        allowed_providers: set[str],
        per_provider_limit: int,
    ) -> list[ResearchSource]:
        provider_searchers = self._provider_searchers()
        sources: list[ResearchSource] = []
        routed_providers = [
            provider
            for provider in self._provider_order(search_query, allowed_providers)
            if provider in allowed_providers
            and not (
                provider == "github-repositories"
                and not self._looks_like_software_agent_query(search_query)
            )
            and provider_searchers.get(provider) is not None
        ]
        parallel_workers = self._provider_parallel_worker_count(
            len(routed_providers)
        )
        if getattr(
            self._provider_parallelism_context,
            "disable_nested_provider_parallelism",
            False,
        ):
            parallel_workers = 1

        if parallel_workers > 1 and len(routed_providers) > 1:
            return self._run_provider_dispatch_async(
                routed_providers,
                provider_searchers,
                search_query,
                per_provider_limit,
            )

        for provider in routed_providers:
            sources.extend(
                self._search_provider_results(
                    provider,
                    provider_searchers[provider],
                    search_query,
                    per_provider_limit,
                )
            )
        return sources

    def _run_provider_dispatch_async(
        self,
        routed_providers: list[str],
        provider_searchers: dict[str, Any],
        search_query: str,
        per_provider_limit: int,
    ) -> list[ResearchSource]:
        async def _dispatch() -> list[ResearchSource]:
            provider_results_by_name: dict[str, list[ResearchSource]] = {}
            try:
                import httpx  # type: ignore[import-not-found]
            except ImportError:
                httpx = None

            if httpx is None:
                for provider in routed_providers:
                    provider_results_by_name[provider] = await asyncio.to_thread(
                        self._search_provider_results,
                        provider,
                        provider_searchers[provider],
                        search_query,
                        per_provider_limit,
                    )
            else:
                try:
                    import h2  # noqa: F401  # type: ignore[import-not-found]
                    _http2_dispatch = True
                except ImportError:
                    _http2_dispatch = False

                async with httpx.AsyncClient(
                    follow_redirects=True,
                    http2=_http2_dispatch,
                    verify=True,
                ) as client:
                    tasks = [
                        self._search_provider_results_async(
                            provider,
                            provider_searchers[provider],
                            search_query,
                            per_provider_limit,
                            client,
                        )
                        for provider in routed_providers
                    ]
                    for name, provider_results in await asyncio.gather(*tasks):
                        provider_results_by_name[name] = provider_results

            sources: list[ResearchSource] = []
            for provider in routed_providers:
                sources.extend(provider_results_by_name.get(provider) or [])
            return sources

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_dispatch())

        container: dict[str, Any] = {}

        def _runner() -> None:
            container["value"] = asyncio.run(_dispatch())

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        return list(container.get("value") or [])

    async def _search_provider_results_async(
        self,
        provider: str,
        searcher: Any,
        search_query: str,
        per_provider_limit: int,
        client: Any,
    ) -> tuple[str, list[ResearchSource]]:
        limit = (
            min(per_provider_limit, 5)
            if provider == "github-repositories"
            else per_provider_limit
        )
        async_searcher = self._provider_async_searchers().get(provider)
        if async_searcher is None:
            return (
                provider,
                await asyncio.to_thread(
                    self._search_provider_results,
                    provider,
                    searcher,
                    search_query,
                    per_provider_limit,
                ),
            )
        try:
            provider_results = await async_searcher(
                search_query,
                limit,
                client,
            )
        except Exception as exc:
            self._record_provider_diagnostic(
                provider,
                "query-error",
                f"{type(exc).__name__}: {exc}",
            )
            return provider, []
        return provider, provider_results

    def _search_provider_results(
        self,
        provider: str,
        searcher: Any,
        search_query: str,
        per_provider_limit: int,
    ) -> list[ResearchSource]:
        limit = (
            min(per_provider_limit, 5)
            if provider == "github-repositories"
            else per_provider_limit
        )
        try:
            provider_results = searcher(search_query, limit)
        except Exception as exc:
            self._record_provider_diagnostic(
                provider,
                "query-error",
                f"{type(exc).__name__}: {exc}",
            )
            return []
        if not provider_results:
            self._record_provider_diagnostic(
                provider,
                "query-empty",
                f"0 results for query: {search_query[:120]}",
            )
            return []
        return provider_results

    @classmethod
    def _provider_order(
        cls,
        search_query: str = "",
        allowed_providers: set[str] | None = None,
    ) -> tuple[str, ...]:
        return _provider_order_policy(
            search_query,
            allowed_providers,
            looks_like_software_agent_query=cls._looks_like_software_agent_query,
            looks_like_market_query=cls._looks_like_market_query,
            looks_like_academic_query=cls._looks_like_academic_query,
            looks_like_current_evidence_query=cls._looks_like_current_evidence_query,
        )

    def _provider_searchers(self) -> dict[str, Any]:
        return {
            "sec-edgar": self._search_sec_edgar,
            "financial-portals": self._search_financial_portals,
            "earnings-data": self._search_earnings_data,
            "insider-transactions": self._search_insider_transactions,
            "short-interest": self._search_short_interest,
            "macrotrends": self._search_macrotrends,
            "stockanalysis": self._search_stockanalysis,
            "fed-macro": self._search_fed_macro_data,
            "google-news-rss": self._search_google_news_rss,
            "seeking-alpha": self._search_seeking_alpha_news,
            "reddit-finance": self._search_reddit_finance,
            "bing-search": self._search_bing_results,
            "openalex": self._search_openalex,
            "semantic-scholar": self._search_semantic_scholar,
            "crossref": self._search_crossref,
            "web-search": self._search_web_results,
            "github-repositories": self._search_github_repositories,
        }

    def _provider_async_searchers(self) -> dict[str, Any]:
        searchers: dict[str, Any] = {}
        for provider, name in (
            ("openalex", "_search_openalex_async"),
            ("semantic-scholar", "_search_semantic_scholar_async"),
            ("crossref", "_search_crossref_async"),
        ):
            searcher = getattr(self, name, None)
            if callable(searcher):
                searchers[provider] = searcher
        return searchers
