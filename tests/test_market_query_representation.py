from __future__ import annotations

from agentos_orchestrator.research.deep_research import DeepResearchEngine


def test_query_from_objective_prefers_substantive_stock_clause() -> None:
    objective = (
        "tune provider-side recall for broad live market discovery and rerun "
        "the exact end-to-end stock report. run a full analyst-style report "
        "generation on one concrete ticker/theme and assess final report "
        "quality directly. highest potential probability-adjusted upside among "
        "publicly traded companies over the next 12 months"
    )

    query = DeepResearchEngine._query_from_objective(objective)

    assert "publicly traded companies" in query
    assert "provider-side recall" not in query
    assert "analyst-style" not in query


def test_market_query_anchors_promote_constraints_not_generic_comparatives() -> None:
    query = (
        "highest potential probability-adjusted upside among publicly traded "
        "companies over the next 12 months"
    )

    anchors = DeepResearchEngine._objective_anchor_terms(query)

    assert "highest" not in anchors
    assert "potential" not in anchors
    assert "upside" not in anchors
    assert "publicly traded" in anchors
    assert "public company" in anchors
    assert "earnings" in anchors
    assert "price target" in anchors


def test_market_query_alignment_rejects_lexical_false_positive() -> None:
    query = (
        "highest potential probability-adjusted upside among publicly traded "
        "companies over the next 12 months"
    )
    story_text = (
        "ArcGIS StoryMaps travel upside guide for public art in listed buildings"
    )
    market_text = (
        "Analyst price target revisions and earnings guidance for publicly "
        "traded semiconductor stocks"
    )

    assert DeepResearchEngine._objective_alignment_score(story_text, query) == 0.0
    assert DeepResearchEngine._objective_alignment_score(market_text, query) >= 0.75
