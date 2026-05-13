from __future__ import annotations

import json
import os
import re
import urllib.parse
from itertools import combinations
from pathlib import Path
from typing import Any

from .models import (
    ResearchSource,
    extract_ticker_candidates as _extract_ticker_candidates,
)
from .query_policy import (
    blocked_irrelevant_site_hosts as _blocked_irrelevant_site_hosts_policy,
    blocked_market_query_site_hosts as _blocked_market_query_site_hosts_policy,
    has_query_scaffold_noise as _has_query_scaffold_noise_policy,
    normalize_research_plan_query as _normalize_research_plan_query_policy,
    trim_query_variant_text as _trim_query_variant_text_policy,
)


class ResearchPlanningSynthesisMixin:
    @staticmethod
    def _evidence_grade_rank(evidence_grade: str) -> int:
        return {
            "strong": 4,
            "tool-observation": 3,
            "moderate": 2,
            "weak": 1,
            "ungraded": 0,
        }.get(evidence_grade, 0)

    @staticmethod
    def _finding_confidence_rank(confidence: str) -> int:
        return {
            "high": 4,
            "medium": 3,
            "low": 2,
            "needs-verification": 1,
        }.get(confidence, 0)

    def _brief_markdown(
        self,
        objective: str,
        query: str,
        summary: str,
        sources: list[ResearchSource],
        depth: str,
    ) -> str:
        lines = [
            "# Deep Research Brief",
            "",
            f"Objective: {objective}",
            "",
            f"Depth: {depth}",
            "",
            f"Query: {query}",
            "",
            "## Synthesis",
            "",
            summary,
            "",
            "## Evidence Quality",
            "",
            self._quality_summary(sources),
            "",
            "## Sources",
            "",
        ]
        for index, source in enumerate(sources, start=1):
            authors = ", ".join(source.authors[:3]) or "Unknown authors"
            year = source.year or "n.d."
            lines.extend(
                [
                    f"{index}. {source.title}",
                    f"   Provider: {source.provider}",
                    f"   Authors: {authors}",
                    f"   Year: {year}",
                    f"   Grade: {source.evidence_grade}",
                    (
                        "   Quality: "
                        f"relevance {source.relevance:.2f}, "
                        f"recency {source.recency:.2f}, "
                        f"citations {source.citation_strength:.2f}, "
                        f"contradiction risk {source.contradiction_risk:.2f}"
                    ),
                    f"   URL: <{source.url}>",
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def _synthesis_source_limit(
        cls,
        depth: str,
        source_count: int,
        synthesis_mode: str,
        objective: str = "",
    ) -> int:
        if source_count <= 0:
            return 0
        if synthesis_mode == "durable-notes-only":
            return min(source_count, 24)
        limits = {
            "quick": 12,
            "standard": 24,
            "multi-hour": 32,
        }
        if (
            depth == "multi-hour"
            and objective
            and (
                cls._looks_like_software_agent_query(objective)
                or cls._looks_like_current_evidence_query(objective)
                or cls._looks_like_comprehensive_research(objective.lower())
            )
        ):
            return max(1, min(source_count, 96))
        return max(1, min(source_count, limits.get(depth, 24)))

    def _build_synthesis_packet(
        self,
        objective: str,
        query: str,
        sources: list[ResearchSource],
        depth: str,
        plan: dict[str, Any] | None,
        durable_notes: str,
        synthesis_mode: str,
    ) -> dict[str, Any]:
        source_limit = self._synthesis_source_limit(
            depth,
            len(sources),
            synthesis_mode,
            objective,
        )
        synthesis_sources = sources[:source_limit]
        findings = self._finding_ledger(query or objective, synthesis_sources, plan)
        perspective_coverage = self._perspective_coverage(
            sources,
            (plan or {}).get("perspectives") or [],
        )
        provider_counts: dict[str, int] = {}
        for source in sources:
            provider_counts[source.provider] = (
                provider_counts.get(source.provider, 0) + 1
            )
        return {
            "objective": objective,
            "query": query,
            "depth": depth,
            "synthesis_mode": synthesis_mode,
            "total_ranked_sources": len(sources),
            "synthesis_source_count": len(synthesis_sources),
            "durable_notes_available": bool(durable_notes.strip()),
            "provider_counts": provider_counts,
            "perspective_coverage": perspective_coverage,
            "findings": findings,
            "market_signals": (
                self._market_signal_snapshot(synthesis_sources)
                if self._looks_like_market_query(query or objective)
                else []
            ),
            "top_sources": [
                {
                    "provider": source.provider,
                    "title": source.title,
                    "url": source.url,
                    "year": source.year,
                    "evidence_grade": source.evidence_grade,
                    "score": round(source.score, 3),
                    "abstract": (source.abstract or source.title)[:240],
                }
                for source in synthesis_sources
            ],
        }

    def _summarize(
        self,
        objective: str,
        sources: list[ResearchSource],
        depth: str = "standard",
        plan: dict[str, Any] | None = None,
        query: str = "",
        durable_notes: str = "",
        synthesis_mode: str = "hybrid",
        synthesis_packet: dict[str, Any] | None = None,
    ) -> str:
        if not sources:
            return (
                "Live research did not return sources from configured public "
                f"providers for: {objective}. Check network policy, API "
                "availability, or attach MCP research servers."
            )
        packet = synthesis_packet or self._build_synthesis_packet(
            objective,
            query,
            sources,
            depth,
            plan,
            durable_notes,
            synthesis_mode,
        )
        synthesis_source_count = int(packet.get("synthesis_source_count") or 0)
        synthesis_sources = (
            sources[:synthesis_source_count] if synthesis_source_count > 0 else sources
        )
        if synthesis_mode == "durable-notes-only" and durable_notes.strip():
            ai_synthesis = self._ai_durable_notes_synthesis(
                objective,
                synthesis_sources,
                durable_notes,
                plan,
            )
            if ai_synthesis:
                return ai_synthesis
            note_lines = [
                line
                for line in durable_notes.splitlines()
                if line.strip().startswith("- [")
            ]
            return (
                f"Durable-notes synthesis mode used for: {objective}. "
                f"Integrated {len(note_lines)} distilled claims from workflows/report.md "
                "with minimal source metadata."
            )
        findings = list(packet.get("findings") or [])
        perspective_coverage = dict(packet.get("perspective_coverage") or {})
        subquestion_count = len((plan or {}).get("subquestions", []))

        # ------------------------------------------------------------------
        # SCIENTIST SYNTHESIS: ask AI to reason about what the evidence
        # actually says — forming conclusions, noting contradictions, and
        # flagging gaps — rather than generating a boilerplate summary.
        # ------------------------------------------------------------------
        ai_synthesis = self._ai_scientist_synthesis(
            objective,
            synthesis_sources,
            findings,
            plan,
            perspective_coverage,
            durable_notes,
            synthesis_mode,
        )
        if ai_synthesis:
            return ai_synthesis

        # Fallback: produce a structured analyst-grade report from raw sources
        # when AI synthesis is unavailable (no Gemini API key configured).
        return self._structured_fallback_report(
            objective=objective,
            sources=sources,
            findings=findings,
            plan=plan,
            perspective_coverage=perspective_coverage,
            market_signal_lines=list(packet.get("market_signals") or []),
            depth=depth,
            durable_notes=durable_notes,
        )

    def _structured_fallback_report(
        self,
        objective: str,
        sources: list[ResearchSource],
        findings: list[dict[str, Any]],
        plan: dict[str, Any] | None,
        perspective_coverage: dict[str, Any],
        market_signal_lines: list[str],
        depth: str,
        durable_notes: str = "",
    ) -> str:
        """Generate a structured analyst-style report from raw sources.

        This runs when no AI key is configured. It extracts all material
        evidence from source abstracts, durable notes, and market signals into
        a readable, structured markdown report — not a generic template string.
        """
        lines: list[str] = []
        market_mode = self._looks_like_market_query(objective)

        lines.append(f"# Research Report: {objective}")
        lines.append("")
        lines.append(
            f"**Depth**: {depth} | **Sources**: {len(sources)} | "
            f"**Perspectives covered**: "
            f"{perspective_coverage.get('count', 0)}/"
            f"{perspective_coverage.get('total', 0)}"
        )
        lines.append("")

        # Executive summary from durable notes (best quality content)
        if durable_notes and len(durable_notes.strip()) > 200:
            lines.append("## Key Findings (Distilled from Deep Research)")
            lines.append("")
            # Extract bullet points from durable notes
            note_bullets = [
                ln.strip()
                for ln in durable_notes.splitlines()
                if ln.strip().startswith("- [") or ln.strip().startswith("- **")
            ][:25]
            if note_bullets:
                for bullet in note_bullets:
                    lines.append(bullet)
                lines.append("")
            else:
                # Fallback: take first 2000 chars of durable notes
                lines.append(durable_notes[:2000])
                lines.append("")

        # Market signals / ticker candidates
        if market_mode and market_signal_lines:
            lines.append("## Market Candidates Identified")
            lines.append("")
            lines.append(
                "The following companies/tickers were identified as candidates "
                "meeting the stated criteria:"
            )
            lines.append("")
            for sig in market_signal_lines[:20]:
                lines.append(f"- {sig}")
            lines.append("")

        # Per-perspective findings
        if findings:
            lines.append("## Evidence by Research Angle")
            lines.append("")
            for f in findings[:8]:
                perspective = f.get("perspective", "General")
                finding_text = f.get("finding", "")
                confidence = f.get("confidence", "unknown")
                n_sources = f.get("support_count", 0)
                contradictions = f.get("contradiction_count", 0)
                lines.append(
                    f"### {perspective} [confidence: {confidence}, "
                    f"{n_sources} sources"
                    + (f", {contradictions} contradictions" if contradictions else "")
                    + "]"
                )
                lines.append("")
                lines.append(finding_text)
                lines.append("")

        # Source evidence table
        if sources:
            lines.append("## Source Evidence")
            lines.append("")
            top_sources = sorted(
                sources, key=lambda s: float(s.score or 0), reverse=True
            )[:20]
            for i, src in enumerate(top_sources, 1):
                title = src.title or "Untitled"
                url = src.url or ""
                provider = src.provider or "unknown"
                abstract = (src.abstract or "").strip()[:500]
                year = f" ({src.year})" if src.year else ""
                link = f"[{title}]({url})" if url else title
                lines.append(f"**{i}. {link}** [{provider}]{year}")
                if abstract:
                    lines.append(f"> {abstract}")
                lines.append("")

        # Missing coverage
        missing = perspective_coverage.get("missing") or []
        if missing:
            lines.append("## Research Gaps")
            lines.append("")
            lines.append(
                "The following perspectives were not covered by available sources "
                "and warrant additional investigation:"
            )
            lines.append("")
            for m in missing[:6]:
                lines.append(f"- {m}")
            lines.append("")

        return "\n".join(lines)

    def _ai_scientist_synthesis(
        self,
        objective: str,
        sources: list[ResearchSource],
        findings: list[dict[str, Any]],
        plan: dict[str, Any] | None,
        perspective_coverage: dict[str, Any],
        durable_notes: str = "",
        synthesis_mode: str = "hybrid",
    ) -> str:
        """Synthesize evidence like a Wall Street analyst (for market queries)
        or a senior research scientist (for all other topics).

        Wall Street mode produces: investment thesis, bull/bear/base case with
        probability weights, named tickers, catalysts with dates, risk factors
        ranked by severity, comp valuation, and a conviction call.

        Science mode produces: evidence weighing, contradiction analysis,
        credibility grading, gap identification, and confidence calibration.

        Returns an empty string when AI is unavailable.
        """
        if not sources:
            return ""
        market_mode = self._looks_like_market_query(objective)

        # Build compact evidence digest.
        source_lines: list[str]
        if synthesis_mode == "durable-notes-only":
            source_lines = self._minimal_source_metadata_lines(sources)
        else:
            source_lines = []
            for src in sources:
                year = f" ({src.year})" if src.year else ""
                grade = src.evidence_grade
                snippet = (src.abstract or src.title)[:200].replace("\n", " ")
                source_lines.append(
                    f"[{src.provider}/{grade}] {src.title}{year}: {snippet}"
                )

        finding_lines: list[str] = []
        for f in findings[:10]:
            finding_lines.append(
                f"  {f['perspective']} ({f['confidence']}, "
                f"{f['support_count']} sources, "
                f"{f['contradiction_count']} contradictions): {f['finding']}"
            )
        missing = ", ".join(perspective_coverage.get("missing") or []) or "none"
        market_signal_lines = (
            self._market_signal_snapshot(sources) if market_mode else []
        )
        subquestions = (
            "\n".join(f"  - {sq}" for sq in (plan or {}).get("subquestions", [])[:10])
            or "  (none recorded)"
        )

        if market_mode:
            system = (
                "You are a Managing Director at a top-tier Wall Street investment bank "
                "(think Goldman Sachs, Morgan Stanley, JPMorgan). You are writing a "
                "research note that will be distributed to institutional investors. "
                "This is NOT a summary — it is a rigorous investment analysis.\n\n"
                "YOUR ANALYSIS MUST INCLUDE ALL OF THE FOLLOWING:\n\n"
                "1. EXECUTIVE SUMMARY & RECOMMENDATION (1 paragraph)\n"
                "   - Clear Buy / Overweight / Hold / Underweight / Sell call\n"
                "   - Conviction level: High / Medium / Low (with justification)\n"
                "   - 12-month price target with upside/downside % vs current price\n"
                "   - One-sentence thesis statement\n\n"
                "2. INVESTMENT THESIS — BULL CASE (probability weight: X%)\n"
                "   - Specific catalysts that would drive the thesis\n"
                "   - Valuation in upside scenario (P/E, EV/EBITDA, or P/S)\n"
                "   - Key data points supporting this case with sources cited\n\n"
                "3. INVESTMENT THESIS — BASE CASE (probability weight: X%)\n"
                "   - Expected trajectory with specific metrics (revenue growth, margin)\n"
                "   - Fair value under consensus assumptions\n"
                "   - Upcoming catalysts and their expected impact\n\n"
                "4. INVESTMENT THESIS — BEAR CASE (probability weight: X%)\n"
                "   - Specific risks that would derail the thesis\n"
                "   - Downside scenario valuation\n"
                "   - Counterarguments and short thesis points\n\n"
                "5. CATALYST CALENDAR\n"
                "   - List specific upcoming events with dates/quarters "
                "(earnings, product launches, regulatory decisions, management events)\n"
                "   - Rate each catalyst: positive / negative / binary\n\n"
                "6. COMPARABLE COMPANY ANALYSIS\n"
                "   - Name at least 3-5 comparable companies WITH TICKERS\n"
                "   - Compare key multiples (P/E, EV/EBITDA, P/S, EV/Revenue)\n"
                "   - State whether target is cheap, fairly valued, or expensive vs comps\n\n"
                "7. KEY RISKS (ranked by severity)\n"
                "   - At least 5 specific risks with quantified potential impact\n"
                "   - Include: competitive risk, regulatory risk, macro risk, "
                "execution risk, balance sheet risk\n\n"
                "8. EVIDENCE QUALITY ASSESSMENT\n"
                "   - Which findings are supported by primary sources "
                "(SEC filings, earnings transcripts, official data)\n"
                "   - Which are based on secondary sources (analyst reports, news)\n"
                "   - Key uncertainties that require more diligence\n\n"
                "CRITICAL RULES:\n"
                "- ALWAYS name specific companies with their ticker symbols in parentheses\n"
                "- ALWAYS use specific numbers (%, $, multiples) not vague language\n"
                "- NEVER say 'the company' without naming it\n"
                "- NEVER use phrases like 'it is clear that' or 'obviously'\n"
                "- If evidence is thin on a section, say so explicitly — do not fabricate\n"
                "- Bull/bear/base probabilities must sum to 100%"
            )
            user = (
                f"Research objective: {objective}\n\n"
                f"Subquestions investigated:\n{subquestions}\n\n"
                f"Evidence collected ({len(sources)} sources across "
                f"{len({s.provider for s in sources})} providers):\n"
                + "\n".join(source_lines)
                + (
                    "\n\nMarket signal candidates identified in evidence:\n"
                    + "\n".join(market_signal_lines)
                    if market_signal_lines
                    else ""
                )
                + (
                    "\n\nDurable distilled report notes (deep research accumulation):\n"
                    + durable_notes[:16000]
                    if durable_notes
                    else ""
                )
                + f"\n\nPer-perspective findings:\n"
                + "\n".join(finding_lines or ["  (none yet)"])
                + f"\n\nUncovered perspectives: {missing}\n\n"
                + "Write the complete Wall Street research note as specified above. "
                + "Be forensically specific — institutional investors will scrutinize "
                + "every number and claim. Cite the evidence type for each major assertion."
            )
        else:
            system = (
                "You are a senior research scientist writing an evidence synthesis "
                "for a peer-reviewed audience. Your task is NOT to summarize — "
                "it is to ANALYZE with the rigor of a Nature or Science Methods paper.\n\n"
                "YOUR ANALYSIS MUST:\n"
                "1. FORM EXPLICIT CONCLUSIONS with stated confidence levels "
                "(high/moderate/low/speculative) and the minimum evidence "
                "threshold that would change each conclusion.\n"
                "2. WEIGH CONTRADICTIONS forensically: for every conflicting claim, "
                "identify whether the cause is methodological, scope-related, "
                "recency bias, or sampling artifact — and state which side the "
                "weight of evidence favors and why.\n"
                "3. GRADE SOURCE CREDIBILITY for every major claim: "
                "peer-reviewed > pre-registered > government data > industry reports "
                "> preprints > blogs > uncited claims. Flag over-reliance on "
                "low-credibility sources explicitly.\n"
                "4. MAP CAUSAL MECHANISMS: don't just state what was found — "
                "explain the mechanism. What drives the effect? What are the "
                "confounders? What are the effect sizes?\n"
                "5. IDENTIFY CRITICAL GAPS: name exactly what evidence is missing, "
                "why it matters, and what kind of study would fill it.\n"
                "6. ASSESS REPLICATION STATUS: has the finding been independently "
                "replicated? Are there failed replications? What is the p-curve?\n"
                "7. PRACTICAL IMPLICATIONS: what do the findings mean for real-world "
                "application? What are the scope conditions and boundary cases?\n\n"
                "Be specific, technical, and never use filler language. "
                "A reader should be able to cite your analysis in a paper."
            )
            user = (
                f"Research objective: {objective}\n\n"
                f"Subquestions investigated:\n{subquestions}\n\n"
                f"Evidence found ({len(sources)} sources):\n"
                + "\n".join(source_lines)
                + (
                    "\n\nDurable distilled report notes:\n" + durable_notes[:16000]
                    if durable_notes
                    else ""
                )
                + f"\n\nPer-perspective findings:\n"
                + "\n".join(finding_lines or ["  (none yet)"])
                + f"\n\nUncovered perspectives: {missing}\n\n"
                + "Synthesize this evidence as a senior research scientist would. "
                + "Be substantive, specific, and technically rigorous. "
                + "A reader must be able to make informed decisions or design "
                + "follow-up studies based on your synthesis."
            )
        return self._call_ai_text(system, user)

    def _ai_durable_notes_synthesis(
        self,
        objective: str,
        sources: list[ResearchSource],
        durable_notes: str,
        plan: dict[str, Any] | None,
    ) -> str:
        """Synthesize using only durable report notes plus minimal metadata."""
        if not durable_notes.strip():
            return ""
        metadata_lines = "\n".join(self._minimal_source_metadata_lines(sources))
        subquestions = (
            "\n".join(f"  - {sq}" for sq in (plan or {}).get("subquestions", [])[:6])
            or "  (none recorded)"
        )
        system = (
            "You are a senior research scientist. Build the final synthesis using "
            "ONLY the provided durable notes and minimal source metadata. "
            "Do not request or infer hidden abstract text. "
            "Explicitly weigh contradictions, confidence, and missing evidence."
        )
        user = (
            f"Research objective: {objective}\n\n"
            f"Subquestions investigated:\n{subquestions}\n\n"
            "Durable report notes:\n"
            f"{durable_notes[:16000]}\n\n"
            "Minimal source metadata:\n"
            f"{metadata_lines}\n\n"
            "Write a 4-8 paragraph final synthesis with confidence levels and "
            "contradiction analysis."
        )
        return self._call_ai_text(system, user)

    @staticmethod
    def _minimal_source_metadata_lines(sources: list[ResearchSource]) -> list[str]:
        lines: list[str] = []
        for src in sources[:40]:
            year = str(src.year) if src.year else "n.d."
            lines.append(
                (
                    f"- [{src.provider}/{src.evidence_grade}] {src.title} "
                    f"(year: {year}) url: {src.url}"
                )[:320]
            )
        return lines

    @staticmethod
    def _resolve_final_synthesis_mode(depth: str, durable_notes: str) -> str:
        configured = (
            str(os.environ.get("AGENTOS_FINAL_SYNTHESIS_MODE") or "").strip().lower()
        )
        if configured in {"hybrid", "durable-notes-only"}:
            return configured
        return "hybrid"

    def _initialize_durable_report(
        self,
        run_id: str,
        depth: str,
        objective: str,
    ) -> Path | None:
        if not run_id:
            return None
        report_path = self._durable_report_path(run_id)
        if report_path is None:
            return None
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not report_path.exists():
            report_path.write_text(
                (
                    "# Durable Research Report\n\n"
                    f"Depth: {depth}\n\n"
                    f"Objective: {objective}\n\n"
                    "## Incremental Findings\n\n"
                ),
                encoding="utf-8",
            )
        else:
            try:
                existing = report_path.read_text(encoding="utf-8")
                self._durable_note_passes = {
                    int(match.group(1))
                    for match in re.finditer(r"^###\s+Pass\s+(\d+)\b", existing, re.M)
                }
            except (OSError, ValueError):
                self._durable_note_passes = set()
        return report_path

    def _append_durable_claim_notes(
        self,
        report_path: Path | None,
        pass_index: int,
        sources: list[ResearchSource],
        query: str,
    ) -> None:
        if report_path is None or not sources:
            return
        if pass_index in self._durable_note_passes:
            return
        lines: list[str] = [f"### Pass {pass_index}"]
        wrote_any = False
        for source in sources:
            if not source.url or source.url in self._durable_note_urls:
                continue
            if source.evidence_grade not in {"strong", "moderate", "tool-observation"}:
                continue
            claim = self._compressed_claim(source, query)
            if not claim:
                continue
            wrote_any = True
            self._durable_note_urls.add(source.url)
            lines.append(
                "- "
                f"[{source.evidence_grade}/{source.provider}] {claim} "
                f"(source: {source.url})"
            )
        if not wrote_any:
            lines.append("- [info/system] no-new-distilled-claims this pass")
        lines.append("")
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self._durable_note_passes.add(pass_index)

    @classmethod
    def _compressed_claim(cls, source: ResearchSource, query: str = "") -> str:
        text = (source.abstract or source.title or "").strip()
        if not text:
            return ""
        lower_raw = text.lower()
        if (
            query
            and cls._looks_like_market_query(query)
            and source.provider == "gemini-flash"
        ):
            return ""
        if query and cls._looks_like_market_query(query):
            tickers = _extract_ticker_candidates(f"{source.title} {source.abstract}")
            if len(tickers) >= 2:
                return f"Ticker candidates mentioned: {', '.join(tickers)}"
        if (
            lower_raw.startswith("generic web result")
            or "snippet unavailable" in lower_raw
        ):
            return ""
        text = cls._html_to_text(text)
        if re.search(r"\{.*\}|\[.*\]|\"[a-z0-9_-]+\"\s*:\s*", text[:500], re.I):
            return ""
        text = re.sub(
            r"\b(?:jats|xml|xmlns|sec-type|content-type|article-meta)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip()
        if not text or cls._text_signal_score(text) < 0.07:
            return ""
        if len(re.findall(r"[{}\[\]<>]", text)) >= 4:
            return ""
        promotional_markers = (
            "skip to content",
            "top rated",
            "trading signals",
            "subscribe",
            "newsletter",
            "market intelligence",
        )
        if sum(1 for marker in promotional_markers if marker in text.lower()) >= 2:
            return ""
        if query:
            anchors = set(cls._keywords(query)) | set(
                cls._entity_terms_from_query(query)
            )
            if anchors and not any(anchor in text.lower() for anchor in anchors):
                return ""
            if cls._objective_alignment_score(text, query) < 0.22:
                return ""
        sentence = re.split(r"[.!?]", text, maxsplit=1)[0].strip()
        if sentence and cls._text_signal_score(sentence) >= 0.08:
            claim = sentence
        else:
            # Short headline-like first sentences often have low token density.
            # Fall back to the cleaned full text so valid web evidence can still
            # be distilled into durable notes.
            claim = text
        claim = re.sub(r"\s+", " ", claim)
        if cls._text_signal_score(claim) < 0.07:
            return ""
        return claim[:260]

    def _load_durable_notes(self, run_id: str) -> str:
        report_path = self._durable_report_path(run_id)
        if report_path is None or not report_path.exists():
            return ""
        try:
            return report_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _durable_report_path(self, run_id: str) -> Path | None:
        if not run_id:
            return None
        return self.workspace_root / "runs" / run_id / "workflows" / "report.md"

    def _build_research_plan(
        self,
        objective: str,
        query: str,
        depth: str,
        pc_context_info: dict[str, Any],
    ) -> dict[str, Any]:
        # ----------------------------------------------------------------
        # THINKING STEP: Ask AI to reason about this specific objective —
        # what entities are involved, what causal relationships matter, what
        # authoritative sources exist, and what queries would expose the
        # strongest evidence.
        # ----------------------------------------------------------------
        ai_strategy = self._ai_research_strategy(objective, query, depth)
        software_agent_mode = self._looks_like_software_agent_diagnostic_objective(
            f"{objective} {query}"
        )
        diagnostic_queries = self._software_agent_diagnostic_queries(objective)
        diagnostic_perspectives = self._software_agent_diagnostic_perspectives(
            objective
        )
        diagnostic_subquestions = self._software_agent_diagnostic_subquestions(
            objective
        )
        diagnostic_authorities = self._software_agent_diagnostic_seed_urls(objective)

        # ------------------------------------------------------------------
        # ADAPTIVE PLANNING: derive perspectives, comparative axes, and
        # evidence requirements from AI reasoning about THIS specific
        # objective, not from domain-type templates.
        # ------------------------------------------------------------------
        perspectives = (
            diagnostic_perspectives
            if diagnostic_perspectives
            else self._research_perspectives(query, objective, depth)
        )

        ai_axes = self._ai_research_axes(objective, query, depth)
        comparative_axes = ai_axes.get("comparative_axes") or []
        evidence_requirements = ai_axes.get("evidence_requirements") or []
        if software_agent_mode:
            comparative_axes = list(comparative_axes) + [
                "planner/query distillation quality",
                "browser and sandbox tool routing",
                "retrieval breadth and crawl scaling",
                "evidence ranking and synthesis fidelity",
                "benchmark performance against leading research agents",
            ]
            evidence_requirements = list(evidence_requirements) + [
                "code-path evidence for planning, routing, retrieval, and synthesis",
                "browser or sandbox traces showing active tool usage",
                "crawl breadth metrics, queue state, or frontier expansion evidence",
                "authoritative benchmark or product documentation for competitor comparison",
            ]

        # AI-derived subquestions from strategy call above.
        ai_subquestions = list(ai_strategy.get("subquestions") or [])
        if diagnostic_subquestions:
            ai_subquestions = diagnostic_subquestions + ai_subquestions

        if not ai_subquestions:
            # Absolute minimal fallback if AI strategy generation failed.
            ai_subquestions = [
                "What exact problem statement defines the topic?",
                "Which causal drivers and mechanisms recur across the evidence?",
                "What explicit limitations or uncertainties remain?",
            ]

        if not comparative_axes:
            comparative_axes = [
                "source credibility and recency",
                "methodological rigor",
                "stated limitations and uncertainties",
                "independent corroboration",
            ]

        if not evidence_requirements:
            evidence_requirements = [
                "primary or authoritative evidence",
                "explicit causal or methodological data",
                "independent corroboration when available",
                "clear risk, uncertainty, or limitation statements",
            ]

        if pc_context_info.get("browser_context_detected"):
            ai_subquestions.append(
                "How does live browser/app context from the local PC alter the evidence collection sequence?"
            )

        # Deduplicate subquestions (AI-derived only — template_subquestions
        # was removed when planning was made fully AI-first).
        merged_subquestions: list[str] = []
        seen_sq: set[str] = set()
        for sq in list(ai_subquestions):
            key = sq.lower().strip()[:80]
            if key and key not in seen_sq:
                merged_subquestions.append(sq)
                seen_sq.add(key)
        subquestions = merged_subquestions

        # Entity-focused short queries come FIRST so they are not cut by
        # max_query_variants when the list is later sliced.
        plan_queries = list(diagnostic_queries)
        plan_queries.extend(self._entity_queries(query, objective))

        # AI-reasoned queries come next — these are derived from causal
        # thinking, not generic template expansions.
        for rq in ai_strategy.get("reasoning_queries") or []:
            if rq and rq.strip():
                plan_queries.append(self._trim_query_variant_text(rq))
        # Short keyword variants come BEFORE perspectives so that recency /
        # domain-specific variants (e.g. "latest", "2-adic") are not pushed
        # past the max_query_variants cutoff by the larger perspective lists.
        plan_queries.extend(self._query_variants(query, depth))
        for perspective in perspectives:
            plan_queries.extend(perspective.get("queries") or [])
        # Subquestions are turned into short keyword phrases, NOT appended
        # verbatim as full sentence strings (those confuse API search).
        for question in subquestions:
            kw = self._question_to_keywords(question, query)
            if kw:
                plan_queries.append(kw)

        query_reference = query or objective
        normalized_queries: list[str] = []
        for candidate in plan_queries:
            query_text = self._normalize_research_plan_query(
                candidate,
                query_reference,
            )
            if not query_text:
                continue
            normalized_queries.append(query_text)
        deduped_queries = self._sanitize_query_variants(
            normalized_queries,
            query_reference,
        )
        if not deduped_queries:
            fallback_query = self._normalize_research_plan_query(
                query_reference,
                query_reference,
            )
            if fallback_query:
                deduped_queries = [fallback_query]

        merged_domains: list[str] = []
        seen_domains: set[str] = set()
        blocked_domains = self._blocked_irrelevant_site_hosts(query_reference)
        for domain in diagnostic_authorities + list(
            ai_strategy.get("authoritative_domains") or []
        ):
            text = str(domain or "").strip()
            if not text:
                continue
            normalized_url = (
                text
                if re.match(r"^[a-z]+://", text, flags=re.IGNORECASE)
                else f"https://{text.lstrip('/')}"
            )
            if not self._is_safe_public_url(normalized_url):
                continue
            host = urllib.parse.urlparse(normalized_url).netloc.lower().lstrip("www.")
            if any(
                host == blocked or host.endswith(f".{blocked}")
                for blocked in blocked_domains
            ):
                continue
            normalized = normalized_url.rstrip("/").lower()
            if normalized in seen_domains:
                continue
            seen_domains.add(normalized)
            merged_domains.append(normalized_url.rstrip("/"))

        return {
            "core_question": objective[:300],
            "subquestions": subquestions,
            "comparative_axes": comparative_axes,
            "evidence_requirements": evidence_requirements,
            "perspectives": perspectives,
            "query_plan": deduped_queries,
            # AI-reasoned authoritative domains are stored so that
            # _iterative_retrieval can seed the source list with them.
            "ai_authoritative_domains": merged_domains,
            "ai_causal_connections": ai_strategy.get("causal_connections") or [],
        }

    @staticmethod
    def _blocked_market_query_site_hosts() -> set[str]:
        return _blocked_market_query_site_hosts_policy()

    @classmethod
    def _blocked_irrelevant_site_hosts(cls, reference_query: str) -> set[str]:
        return _blocked_irrelevant_site_hosts_policy(
            reference_query,
            looks_like_market_query=cls._looks_like_market_query,
            looks_like_software_agent_query=cls._looks_like_software_agent_query,
        )

    @staticmethod
    def _trim_query_variant_text(candidate: str, limit: int = 240) -> str:
        return _trim_query_variant_text_policy(candidate, limit)

    @staticmethod
    def _has_query_scaffold_noise(text: str) -> bool:
        return _has_query_scaffold_noise_policy(text)

    @classmethod
    def _normalize_research_plan_query(
        cls,
        candidate: str,
        reference_query: str,
    ) -> str:
        return _normalize_research_plan_query_policy(
            candidate,
            reference_query,
            query_core_terms=cls._query_core_terms,
            looks_like_market_query=cls._looks_like_market_query,
            looks_like_software_agent_query=cls._looks_like_software_agent_query,
            is_low_signal_query_variant=cls._is_low_signal_query_variant,
            is_noisy_query_variant=cls._is_noisy_query_variant,
        )

    def _research_perspectives(
        self,
        query: str,
        objective: str,
        depth: str,
    ) -> list[dict[str, Any]]:
        """Generate perspectives for this research objective via AI.

        The ``software_mode`` and ``math_mode`` flags are retained in the
        signature for backward compatibility but are no longer used to select
        a hardcoded list — the AI derives what angles are relevant instead.
        """
        return self._ai_generate_perspectives(query, objective, depth)

    @classmethod
    def _entity_queries(cls, query: str, objective: str) -> list[str]:
        combined = " ".join(part for part in (query, objective) if part).strip()
        lower = combined.lower()
        entities = sorted(
            cls._entity_terms_from_query(combined),
            key=lambda item: (
                lower.find(item.lower())
                if lower.find(item.lower()) >= 0
                else len(lower) + len(item),
                len(item),
            ),
        )
        generic_terms = cls._generic_query_terms()
        entity_tokens = {
            token
            for entity in entities
            for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", entity.lower())
        }

        anchor_tokens: list[str] = []
        seen_anchor_tokens: set[str] = set()
        for token in re.findall(r"\b[a-z][a-z0-9-]{3,}\b", lower):
            if (
                token in generic_terms
                or token in entity_tokens
                or token in seen_anchor_tokens
            ):
                continue
            seen_anchor_tokens.add(token)
            anchor_tokens.append(token)
            if len(anchor_tokens) >= 8:
                break
        for token in cls._keywords(combined):
            if (
                token in generic_terms
                or token in entity_tokens
                or token in seen_anchor_tokens
            ):
                continue
            seen_anchor_tokens.add(token)
            anchor_tokens.append(token)
            if len(anchor_tokens) >= 8:
                break

        focus_phrases: list[str] = []
        if cls._looks_like_current_evidence_query(combined):
            focus_phrases.extend(["latest evidence", "current analysis"])
        if "risk" in lower or "uncertainty" in lower:
            focus_phrases.append("risk analysis")
        if "compare" in lower or len(entities) > 1:
            focus_phrases.append("comparison")

        focused: list[str] = []
        if entities:
            for entity in entities[:4]:
                focused.append(entity)
                for token in anchor_tokens[:4]:
                    focused.append(f"{entity} {token}")
                for phrase in focus_phrases[:3]:
                    if phrase not in entity.lower():
                        focused.append(f"{entity} {phrase}")
            for left, right in combinations(entities[:4], 2):
                focused.append(f"{left} {right} comparison")
        elif cls._looks_like_current_evidence_query(combined):
            core = cls._query_core_terms(combined)
            if core:
                focused.append(core)
                for phrase in focus_phrases[:2]:
                    focused.append(f"{core} {phrase}")

        deduped: list[str] = []
        seen_queries: set[str] = set()
        for candidate in focused:
            text = cls._normalize_research_plan_query(
                candidate[:120],
                combined,
            )
            if not text:
                continue
            normalized = cls._normalize_title(text)
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            deduped.append(text)
        return deduped[:18]

    @classmethod
    def _question_to_keywords(cls, question: str, query: str) -> str:
        """Convert a full subquestion sentence into a short keyword phrase
        suitable for API search (≤60 chars)."""
        # Drop stop words and common filler.
        stop_words = {
            "how",
            "do",
            "does",
            "does",
            "which",
            "what",
            "where",
            "when",
            "are",
            "is",
            "the",
            "a",
            "an",
            "in",
            "of",
            "recently",
            "announced",
            "to",
            "and",
            "or",
            "for",
            "with",
            "from",
            "that",
            "this",
            "their",
            "its",
            "differ",
            "compare",
            "comparisons",
            "vs",
            "system",
            "systems",
        }
        meta_research_terms = {
            "baseline",
            "scope",
            "define",
            "defines",
            "definition",
            "definitions",
            "mechanism",
            "mechanisms",
            "limitation",
            "limitations",
            "failure",
            "mode",
            "modes",
            "causal",
            "factor",
            "factors",
            "driver",
            "drivers",
            "evidence",
            "problem",
            "statement",
            "thesis",
            "theses",
            "topic",
            "exact",
            "explicit",
            "remain",
            "recur",
            "across",
            "uncertainty",
            "uncertainties",
        }
        query_anchors: list[str] = []
        for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", query.lower()):
            if token in stop_words or token in cls._generic_query_terms():
                continue
            if token not in query_anchors:
                query_anchors.append(token)

        words = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", question.lower())
        question_tokens = [w for w in words if w not in stop_words]
        novel_tokens: list[str] = []
        for token in question_tokens:
            if token in cls._generic_query_terms() or token in query_anchors:
                continue
            if token not in novel_tokens:
                novel_tokens.append(token)
        if not novel_tokens:
            return ""
        signal_tokens = [
            token for token in novel_tokens if token not in meta_research_terms
        ]
        if not signal_tokens:
            return ""

        phrase_tokens: list[str] = []
        for token in query_anchors[:4]:
            if token not in phrase_tokens:
                phrase_tokens.append(token)
        for token in signal_tokens:
            if token not in phrase_tokens:
                phrase_tokens.append(token)
            if len(phrase_tokens) >= 6:
                break
        phrase = " ".join(phrase_tokens[:6])
        return phrase[:80].strip() if len(phrase_tokens) >= 3 else ""

    @staticmethod
    def _clean_objective(objective: str) -> str:
        cleaned = re.sub(r"\s+", " ", objective).strip()
        prefixes = (
            "Find authoritative sources, prior systems, and gaps for:",
            "Extract implementation constraints, security boundaries,",
            "Merge worker outputs into a verified research brief for:",
        )
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned

    def _pc_context_summary(
        self,
        pc_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not pc_context:
            return {
                "available": False,
                "snapshot_path": None,
                "node_count": 0,
                "browser_context_detected": False,
                "top_labels": [],
                "judged_site_count": 0,
                "direct_urls": [],
                "discovered_domains": [],
            }

        snapshot_path = Path(str(pc_context.get("snapshot_path") or ""))
        pc_findings = pc_context.get("pc_findings") or {}
        top_labels: list[str] = []
        node_count = 0
        browser_context = False
        if snapshot_path.exists():
            try:
                payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                node_count = len(payload)
                for node in payload:
                    if not isinstance(node, dict):
                        continue
                    name = str(node.get("name") or "").strip()
                    if name:
                        top_labels.append(name)
                    if any(
                        marker in name.lower()
                        for marker in ("browser", "chrome", "edge", "firefox")
                    ):
                        browser_context = True
                    if len(top_labels) >= 8:
                        break
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        direct_urls = [
            url
            for url in self._collect_urls(pc_findings)
            if self._is_safe_public_url(url) and not self._is_search_result_url(url)
        ]
        discovered_domains = [
            str(domain).strip()
            for domain in (pc_findings.get("discovered_domains") or [])
            if str(domain).strip()
        ]
        judged_results = pc_findings.get("judged_results") or []
        if direct_urls or discovered_domains or judged_results:
            browser_context = True
        if not top_labels:
            top_labels = [
                str(label).strip()
                for label in (pc_findings.get("post_snapshot_labels") or [])
                if str(label).strip()
            ][:8]

        return {
            "available": snapshot_path.exists() or bool(pc_findings),
            "snapshot_path": str(snapshot_path).replace("\\", "/"),
            "node_count": node_count,
            "browser_context_detected": browser_context,
            "top_labels": top_labels,
            "judged_site_count": len(judged_results),
            "direct_urls": direct_urls[:6],
            "discovered_domains": discovered_domains[:6],
        }

    def _analysis_report_markdown(
        self,
        objective: str,
        summary: str,
        sources: list[ResearchSource],
        plan: dict[str, Any],
        pc_context_info: dict[str, Any],
    ) -> str:
        lines = [
            "# Deep Research Analysis Report",
            "",
            "## Objective",
            "",
            objective,
            "",
            "## Research Design",
            "",
            f"Core question: {plan['core_question']}",
            "",
            "Subquestions:",
        ]
        for item in plan["subquestions"]:
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "Comparative axes:",
            ]
        )
        for axis in plan["comparative_axes"]:
            lines.append(f"- {axis}")

        lines.extend(
            [
                "",
                "Evidence requirements:",
            ]
        )
        for requirement in plan["evidence_requirements"]:
            lines.append(f"- {requirement}")

        lines.extend(
            [
                "",
                "## Live PC Context",
                "",
                (
                    f"Snapshot available: {pc_context_info['available']}; "
                    f"nodes: {pc_context_info['node_count']}; "
                    "browser context detected: "
                    f"{pc_context_info['browser_context_detected']}"
                ),
                "",
            ]
        )
        if pc_context_info["top_labels"]:
            lines.append("Observed UI labels:")
            for label in pc_context_info["top_labels"]:
                lines.append(f"- {label}")
            lines.append("")

        lines.extend(
            [
                "## Comparative Evidence Matrix",
                "",
                "| Source | Provider | Grade | Key claim |",
                "|---|---|---|---|",
            ]
        )
        for source in sources:
            claim = (source.abstract or source.title).replace("|", " ").strip()
            lines.append(
                "| "
                f"{source.title} | {source.provider} | {source.evidence_grade} | "
                f"{claim[:160]} |"
            )

        lines.extend(
            [
                "",
                "## Synthesis",
                "",
                summary,
                "",
                "## Limitations",
                "",
                "- Provider coverage may vary due to API availability and query drift.",
                "- Repository metadata is not equivalent to peer-reviewed evidence.",
                "- Local PC context was read-only unless explicit act approvals are granted.",
                "",
                "## Next Experiments",
                "",
                "- Run the same plan with controlled query slices per competitor (one system at a time).",
                "- Add explicit benchmark extraction for OSWorld/WebArena task families.",
                "- Add claim-level contradiction checks across providers before final ranking.",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def _query_from_objective(cls, objective: str) -> str:
        cleaned = re.sub(r"\s+", " ", objective).strip()
        prefixes = (
            "Find authoritative sources, direct evidence, and major uncertainties for:",
            "Find authoritative sources, prior systems, and gaps for:",
            "Extract implementation constraints, security boundaries,",
        )
        for prefix in prefixes:
            cleaned = cleaned.replace(prefix, "")
        diagnostic_queries = cls._software_agent_diagnostic_queries(cleaned)
        if diagnostic_queries:
            return diagnostic_queries[0]
        distilled = cls._query_core_terms(cleaned)
        return distilled[:240].strip() or cleaned[:240].strip() or objective[:240]

    @classmethod
    def _software_agent_diagnostic_queries(cls, objective: str) -> list[str]:
        if not cls._looks_like_software_agent_diagnostic_objective(objective):
            return []
        lower = re.sub(r"\s+", " ", objective).strip().lower()
        anchors = ["deep research agent"]
        if "agentos" in lower:
            anchors.insert(0, "agentos deep research")
        elif "orchestrator" in lower:
            anchors.insert(0, "research orchestrator")

        aspects: list[str] = []
        if re.search(
            r"\b(browser|sandbox|pc control|desktop|computer use|local pc|web browsing)\b",
            lower,
        ):
            aspects.append("browser sandbox pc control routing")
        if re.search(
            r"\b(website|websites|url|urls|crawl|crawler|retrieval|breadth|coverage|10k|1000)\b",
            lower,
        ):
            aspects.append("retrieval breadth crawl scaling")
        if re.search(
            r"\b(template|general|generic|useful data|ranking|evidence|synthesis|analyst|scientist)\b",
            lower,
        ):
            aspects.append("evidence quality ranking synthesis")
        if re.search(
            r"\b(compare|comparison|comparable|claude|gpt|gemini|openhands|openclaw)\b",
            lower,
        ):
            aspects.append("benchmark comparison")
        if re.search(
            r"\b(fix|issue|issues|failure|failures|gap|gaps|bug|bugs|why|underperform|shallow)\b",
            lower,
        ):
            aspects.append("failure modes architecture gaps")
        if not aspects:
            aspects = [
                "failure modes architecture gaps",
                "retrieval breadth crawl scaling",
                "evidence quality ranking synthesis",
            ]

        queries: list[str] = []
        comparator_terms = [
            token
            for token in ("claude", "gpt", "gemini", "openhands", "openclaw")
            if token in lower
        ]
        for anchor in anchors:
            queries.append(anchor)
            for aspect in aspects:
                queries.append(f"{anchor} {aspect}")
            for comparator in comparator_terms:
                queries.append(f"{anchor} {comparator} comparison")
        if "mcp" in lower:
            queries.append("model context protocol research agent tool routing")
        if "browser" in lower or "sandbox" in lower:
            queries.append(
                "computer use browser automation research agent architecture"
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            text = str(query or "").strip()[:240]
            normalized = cls._normalize_title(text)
            if not text or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(text)
        return deduped[:14]

    @classmethod
    def _software_agent_diagnostic_seed_urls(cls, objective: str) -> list[str]:
        if not cls._looks_like_software_agent_diagnostic_objective(objective):
            return []
        lower = objective.lower()
        candidates = ["https://github.com"]
        if "mcp" in lower:
            candidates.append("https://modelcontextprotocol.io")
        if "browser" in lower or "sandbox" in lower:
            candidates.append("https://playwright.dev")
        if "claude" in lower:
            candidates.append("https://docs.anthropic.com")
        if "gpt" in lower or "openai" in lower:
            candidates.append("https://platform.openai.com/docs")
        if "gemini" in lower or "google" in lower:
            candidates.append("https://ai.google.dev")

        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped[:8]

    @classmethod
    def _software_agent_diagnostic_perspectives(
        cls,
        objective: str,
    ) -> list[dict[str, Any]]:
        if not cls._looks_like_software_agent_diagnostic_objective(objective):
            return []
        lower = objective.lower()
        perspectives = [
            {
                "name": "architecture",
                "goal": "Audit orchestration, planning, and execution seams.",
                "keywords": [
                    "architecture",
                    "planner",
                    "orchestrator",
                    "routing",
                    "runtime",
                ],
                "queries": [
                    "deep research agent failure modes architecture gaps",
                    "deep research agent planner routing diagnostics",
                ],
            },
            {
                "name": "browser-tooling",
                "goal": "Verify browser, sandbox, and pc-control tool routing.",
                "keywords": [
                    "browser",
                    "sandbox",
                    "pc control",
                    "computer use",
                    "tool routing",
                ],
                "queries": [
                    "deep research agent browser sandbox pc control routing",
                    "computer use browser automation research agent architecture",
                ],
            },
            {
                "name": "retrieval-breadth",
                "goal": "Measure crawl scaling, frontier expansion, and source breadth.",
                "keywords": [
                    "retrieval",
                    "crawl",
                    "breadth",
                    "coverage",
                    "frontier",
                ],
                "queries": [
                    "deep research agent retrieval breadth crawl scaling",
                    "deep research agent frontier expansion coverage",
                ],
            },
            {
                "name": "evidence-quality",
                "goal": "Assess ranking, evidence quality, and synthesis fidelity.",
                "keywords": [
                    "evidence quality",
                    "ranking",
                    "synthesis",
                    "grounding",
                    "useful data",
                ],
                "queries": [
                    "deep research agent evidence quality ranking synthesis",
                    "deep research agent grounding useful data diagnostics",
                ],
            },
        ]
        if re.search(
            r"\b(compare|comparison|comparable|claude|gpt|gemini|openhands|openclaw)\b",
            lower,
        ):
            perspectives.append(
                {
                    "name": "benchmarks",
                    "goal": "Compare observed behavior against leading research agents.",
                    "keywords": [
                        "benchmark",
                        "comparison",
                        "claude",
                        "gpt",
                        "gemini",
                    ],
                    "queries": [
                        "deep research agent benchmark comparison",
                        "deep research agent claude gpt gemini comparison",
                    ],
                }
            )
        return perspectives[:5]

    @classmethod
    def _software_agent_diagnostic_subquestions(cls, objective: str) -> list[str]:
        if not cls._looks_like_software_agent_diagnostic_objective(objective):
            return []
        questions = [
            "Which planning or query-distillation steps are causing low-signal retrieval?",
            "Which browser, sandbox, or pc-control routing decisions are limiting active research?",
            "Where do retrieval, ranking, or synthesis stages discard high-signal evidence?",
            "Which benchmark or competitor capabilities reveal the largest architectural gaps?",
        ]
        if "10k" in objective.lower() or "1000" in objective.lower():
            questions.append(
                "Which crawl, frontier, or queue limits prevent broad multi-thousand URL coverage?"
            )
        return questions[:5]

    @classmethod
    def _split_depth(cls, objective: str) -> tuple[str, str]:
        match = re.search(r"\[(quick|standard|multi-hour|adaptive)\]\s*", objective)
        if match is None:
            cleaned = objective.strip()
            return cls.adaptive_depth_for_objective(cleaned), cleaned
        cleaned = f"{objective[: match.start()]}{objective[match.end() :]}".strip()
        marker = match.group(1)
        if marker == "adaptive":
            return cls.adaptive_depth_for_objective(cleaned), cleaned
        return marker, cleaned

    @classmethod
    def research_depth_for_objective(cls, objective: str) -> str:
        depth, _cleaned = cls._split_depth(objective)
        return depth

    @classmethod
    def adaptive_depth_for_objective(cls, objective: str) -> str:
        """Infer research effort from task complexity using AI.

        Analyzes the objective to decide if it needs a quick lookup,
        standard research, or multi-hour deep investigation.
        """
        system = (
            "You are a research effort estimator. Analyze the research objective "
            "and decide which depth category it requires:\n"
            "- 'quick': simple lookups, single facts, or basic definitions.\n"
            "- 'standard': topics requiring cross-referencing multiple sources "
            "or basic market/technical analysis.\n"
            "- 'multi-hour': deep scientific, financial, or academic research "
            "requiring citation chasing, evidence weighing, and exhaustive foraging.\n"
            "Respond ONLY with one of the three strings: quick, standard, multi-hour."
        )
        try:
            # We use a static-like call here; in practice the orchestrator
            # would pass a client.
            raw = cls()._call_ai_text(system, f"Objective: {objective}")
            raw = raw.lower().strip()
            for depth in ("multi-hour", "standard", "quick"):
                if depth in raw:
                    resolved = depth
                    break
            else:
                resolved = ""
            if resolved:
                lower_objective = objective.lower()
                if resolved == "quick" and cls._looks_like_market_query(
                    lower_objective
                ):
                    if any(
                        cue in lower_objective
                        for cue in (
                            "valuation",
                            "undervalued",
                            "wall street",
                            "ticker",
                            "catalyst",
                            "risk",
                        )
                    ):
                        return "multi-hour"
                    return "standard"
                return resolved
        except Exception as _depth_exc:
            import warnings

            warnings.warn(
                f"AI research-depth classification failed ({_depth_exc!r}); "
                "falling back to heuristic depth classification.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Minimal heuristic fallback if AI is unavailable.
        lower = objective.lower()
        if cls._looks_like_market_query(lower):
            if any(
                cue in lower
                for cue in (
                    "valuation",
                    "undervalued",
                    "wall street",
                    "ticker",
                    "catalyst",
                    "risk",
                    "current evidence",
                )
            ):
                return "multi-hour"
            return "standard"
        if any(
            c in lower for c in ("research", "literature", "systematic", "exhaustive")
        ):
            return "multi-hour"
        if any(c in lower for c in ("compare", "analyze", "benchmark")):
            return "standard"
        return "quick"

    @staticmethod
    def _looks_like_simple_lookup(lower: str) -> bool:
        if len(lower.split()) <= 10 and any(
            cue in lower
            for cue in (
                "recipe",
                "how many",
                "what is",
                "who is",
                "when is",
                "weather",
                "definition",
                "syntax",
                "quick lookup",
            )
        ):
            return True
        return any(
            phrase in lower
            for phrase in (
                "find a recipe",
                "search for a recipe",
                "quick recipe",
                "one source",
                "single source",
            )
        )

    @staticmethod
    def _looks_like_comprehensive_research(lower: str) -> bool:
        comprehensive_cues = (
            "comprehensive",
            "systematic review",
            "scientific literature",
            "literature review",
            "meta-analysis",
            "full report",
            "market report",
            "s&p 500",
            "sp 500",
            "all companies",
            "exhaustive",
            "deep research",
            "state of the art",
            "regulatory landscape",
        )
        if any(cue in lower for cue in comprehensive_cues):
            return True
        return (
            sum(
                1
                for cue in (
                    "compare",
                    "rank",
                    "sources",
                    "evidence",
                    "risks",
                    "limitations",
                    "opportunities",
                    "benchmarks",
                )
                if cue in lower
            )
            >= 3
        )

    @classmethod
    def _query_variants(cls, query: str, depth: str = "standard") -> list[str]:
        """Return query variants generated from objective terms, not fixed templates."""
        ai_variants = cls._ai_query_variants(query, depth)
        if ai_variants:
            return ai_variants

        core = cls._query_core_terms(query)
        if not core:
            return []

        max_variants = 4 if depth == "quick" else 8 if depth == "standard" else 14
        math_mode = cls._looks_like_math_query(query)
        axes = cls._fallback_research_axes(query, depth)
        if math_mode:
            axes = list(dict.fromkeys([*map(str, cls._math_focus_terms(query)), *axes]))

        anchors: list[str] = []
        anchors.extend(sorted(cls._entity_terms_from_query(query)))
        for keyword in sorted(cls._objective_anchor_terms(query)):
            if len(keyword) < 4:
                continue
            if keyword in anchors:
                continue
            anchors.append(keyword)
            if len(anchors) >= 8:
                break
        for keyword in cls._keywords(query):
            if len(keyword) < 4:
                continue
            if keyword in cls._generic_query_terms():
                continue
            if keyword in anchors:
                continue
            anchors.append(keyword)
            if len(anchors) >= 8:
                break
        if math_mode:
            for focus in cls._math_focus_terms(query):
                focus_term = str(focus).strip().lower()
                if not focus_term or focus_term in anchors:
                    continue
                anchors.append(focus_term)
                if len(anchors) >= 12:
                    break
        if core not in anchors:
            anchors.insert(0, core)

        variants: list[str] = [core]
        for anchor in anchors[:4]:
            for axis in axes:
                variants.append(f"{anchor} {axis}")
                if len(variants) >= max_variants * 3:
                    break
            if len(variants) >= max_variants * 3:
                break

        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            candidate = cls._normalize_research_plan_query(variant, query)
            if not candidate:
                continue
            normalized = cls._normalize_title(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(candidate)
            if len(deduped) >= max_variants:
                break
        return deduped

    @classmethod
    def _ai_query_variants(cls, query: str, depth: str = "standard") -> list[str]:
        """Generate search queries using AI.

        For market/current-evidence queries, behaves like a Wall Street analyst
        decomposing an investment thesis into 40-60 targeted searches:
        SEC filings, earnings, comps, catalysts, macro factors, short interest,
        analyst ratings, regulatory filings, technical setups, etc.

        For all other topics, generates academically rigorous sub-questions
        covering methodology, prior work, limitations, and counterevidence.

        Returns an empty list when AI is unavailable so callers fall back to
        the heuristic variant generator.
        """
        if depth == "quick":
            return []
        is_market = cls._looks_like_market_query(query)
        is_current = cls._looks_like_current_evidence_query(query)
        n_queries = 40 if depth == "multi-hour" else 12
        if is_market and (is_current or depth == "multi-hour"):
            system = (
                "You are a senior equity analyst at a top-tier Wall Street firm. "
                "You have been given a research objective and must decompose it into "
                "the exact search queries you would run to build a complete, "
                "publication-quality investment thesis. Think like a real analyst:\n\n"
                "WHAT WALL STREET ACTUALLY RESEARCHES:\n"
                "- Company fundamentals: earnings growth, revenue beats/misses, "
                "margin trajectory, free cash flow, debt/equity, ROE, ROIC\n"
                "- Valuation: P/E, P/S, EV/EBITDA vs sector comps, DCF assumptions, "
                "implied upside to consensus price targets\n"
                "- Catalysts: upcoming earnings dates, product launches, FDA decisions, "
                "contract wins, regulatory approvals, activist events, M&A rumors\n"
                "- Institutional positioning: 13-F filings, insider buying/selling, "
                "short interest %, days-to-cover, options flow\n"
                "- Macro factors: sector rotation, rate sensitivity, FX exposure, "
                "commodity inputs, tariff risk, geopolitical headwinds\n"
                "- Competition: market share trends, moat analysis, competitive threats, "
                "pricing power, customer retention\n"
                "- Bear case: what could go wrong, short thesis, regulatory risk, "
                "management execution risk, leverage concerns\n"
                "- Industry data: TAM growth, channel checks, supply chain dynamics, "
                "inventory levels, demand signals\n"
                "- Technical setup: 52-week range, RSI, moving averages, relative "
                "strength vs index, support/resistance levels\n\n"
                f"Generate exactly {n_queries} specific, targeted search queries "
                "that together would give a complete picture for an investment decision. "
                "Each query must be concrete — include company names, ticker symbols, "
                "specific metrics, and time horizons. Respond ONLY with a JSON array "
                "of strings. No prose, no explanations."
            )
        else:
            system = (
                "You are a senior research scientist. Decompose the research objective "
                "into targeted search queries that together would give a complete, "
                "systematic understanding of the topic. Think like a rigorous scientist:\n\n"
                "WHAT RIGOROUS RESEARCH COVERS:\n"
                "- Primary evidence and mechanism: what actually causes the effect\n"
                "- Methodology: how was it measured, what instruments, what controls\n"
                "- Replication: have independent groups confirmed this\n"
                "- Counterevidence: what contradicts the main finding\n"
                "- Limitations: scope conditions, confounders, publication bias\n"
                "- Recency: what has changed since the original work\n"
                "- Applications: how is it used in practice, what are edge cases\n"
                "- Experts: who are the leading voices, what are their positions\n\n"
                f"Generate exactly {n_queries} specific, targeted search queries. "
                "Each must be concrete and grounded in the actual objective terms. "
                "Respond ONLY with a JSON array of strings. No prose."
            )
        user = f"Research objective: {query}\nDepth: {depth}"
        try:
            raw = cls()._call_ai_text(system, user)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list) and len(parsed) >= 3:
                    result: list[str] = []
                    for item in parsed:
                        candidate = cls._trim_query_variant_text(item)
                        if candidate and len(candidate) >= 6:
                            result.append(candidate)
                    if len(result) >= 3:
                        return result[:n_queries]
        except Exception:
            pass
        return []

    @classmethod
    def _query_core_terms(cls, query: str) -> str:
        """Distill long prompts into domain terms while removing orchestration boilerplate."""
        prefixes = (
            "Perform sandboxed browser research actions for:",
            "Capture live desktop and browser/operator context for:",
            "Find authoritative sources, direct evidence, and major uncertainties for:",
            "Find authoritative sources, prior systems, and gaps for:",
            "Design deep research plan for:",
            "Extract implementation constraints, security boundaries,",
            "Merge worker outputs into a verified research brief for:",
            "Produce a research dossier covering",
            "Produce a rigorous",
        )
        cleaned = re.sub(r"\s+", " ", query).strip()
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()

        cleaned = re.sub(
            (
                r"\busing\s+https?://[^\s<>()]+"
                r"(?:\s+and\s+https?://[^\s<>()]+)*"
                r"\s+as anchor sources\b"
            ),
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"https?://[^\s<>()]+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\b[a-z0-9_./\\-]+\.(?:md|txt|json|ya?ml)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        boilerplate_patterns = (
            r"\bperform(?:ing)? deep research on\b",
            r"\bperform research on\b",
            r"\buse all available [^.;]+",
            r"\busing all available [^.;]+",
            r"\busing all available general-purpose research means[^.;]*",
            r"\bbrowser-grounded web research[^.;]*",
            r"\bsandboxed exploration[^.;]*",
            r"\bcurrent-web evidence[^.;]*",
            r"\bcross-checking\b",
            r"\bbrowser or pc evidence [^.;]+",
            r"\bdurable artifacts\b",
            r"\bproduce (?:a|an) [^.;]+ report\b",
            r"\bdo not use [^.;]+",
            r"\bdo not use (?:a|an) fixed template\b",
            r"\badapt depth and effort [^.;]+",
            r"\baccepted literature\b",
            r"\bplausible proof strategies\b",
            r"\bfocusing on\b",
            r"\bthe exact missing\b",
            r"\bas anchor sources\b",
            r"\bthen expand outward to\b",
            r"\bcorroborat\w*\b",
            r"\banalyst-grade\b",
            r"\bscientist-grade\b",
            r"\branked candidates\b",
            r"\bevidence for and against each thesis[^.;]*",
            r"\bthe evidence for and against each thesis[^.;]*",
            r"\buncertainty bounds\b",
            r"\bcatalyst quality\b",
            r"\bexecution risk\b",
            r"\bvaluation-sensitive considerations\b",
            r"\bclear reasons? for the ranking\b",
            r"\bneed broader domain coverage[^.;]*",
            r"\bneed more direct pages[^.;]*",
            r"\bneed independent verification of browser-derived claims\b",
            r"\bbrowser-derived claims\b",
            r"\bcurrent browser pages\b",
            r"\bsubstantive evidence extraction\b",
        )
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:-")
        if not cleaned:
            return ""

        generic_noise_terms = {
            "generate",
            "analysis",
            "style",
            "explicit",
            "symbols",
            "confidence",
            "levels",
            "as",
            "right",
            "now",
            "research",
            "report",
            "please",
            "could",
            "would",
            "wall",
            "street",
            "based",
            "on",
            "and",
            "for",
            "from",
            "into",
            "with",
            "the",
            "a",
            "an",
            "to",
            "of",
        }
        normalized_tokens = [
            token
            for token in re.findall(r"\b[0-9a-z]+(?:-[0-9a-z]+)*\b", cleaned.lower())
            if token not in generic_noise_terms
        ]
        if cls._looks_like_math_query(query):
            focus_tokens: list[str] = []
            for focus in cls._math_focus_terms(query):
                for token in re.findall(
                    r"\b[0-9a-z]+(?:-[0-9a-z]+)*\b",
                    str(focus).lower(),
                ):
                    if token in generic_noise_terms or token in focus_tokens:
                        continue
                    focus_tokens.append(token)
            if focus_tokens:
                normalized_tokens = focus_tokens + [
                    token for token in normalized_tokens if token not in focus_tokens
                ]
        if len(normalized_tokens) >= 4:
            cleaned = " ".join(normalized_tokens[:18])

        if len(cleaned) > 140:
            stop = {
                "what",
                "which",
                "when",
                "where",
                "why",
                "how",
                "the",
                "a",
                "an",
                "and",
                "or",
                "for",
                "to",
                "with",
                "from",
                "into",
                "about",
                "using",
                "need",
                "please",
                "make",
                "build",
                "create",
                "do",
                "does",
                "did",
                "can",
                "could",
                "should",
                "would",
                "have",
                "has",
                "had",
                "being",
                "been",
            }
            words = [
                token.strip("?.,!:;")
                for token in cleaned.split()
                if token.strip("?.,!:;") and token.strip("?.,!:;").lower() not in stop
            ]
            if words:
                cleaned = " ".join(words)

        return cleaned[:240].strip().lower()
