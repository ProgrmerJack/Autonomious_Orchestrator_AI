"""Tests for programmer_recipes.py (Phase 3)."""

from __future__ import annotations

import textwrap

import pytest

from agentos_orchestrator.cognition.programmer_recipes import (
    build_recipe_code,
    get_recipe,
    list_recipes,
    recipe_for_objective,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Registry                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #


class TestRecipeRegistry:
    def test_all_expected_recipes_registered(self):
        recipes = set(list_recipes())
        expected = {
            "stock_analysis",
            "presentation",
            "pdf_report",
            "docx_report",
            "csv_transform",
            "chart",
            "research_brief",
        }
        for r in expected:
            assert r in recipes, f"Recipe '{r}' not registered"

    def test_get_recipe_returns_none_for_unknown(self):
        assert get_recipe("nonexistent_recipe_xyz_123") is None

    def test_get_recipe_returns_recipe(self):
        recipe = get_recipe("stock_analysis")
        assert recipe is not None
        assert recipe.recipe_id == "stock_analysis"


# ─────────────────────────────────────────────────────────────────────────── #
# recipe_for_objective                                                          #
# ─────────────────────────────────────────────────────────────────────────── #


class TestRecipeForObjective:
    @pytest.mark.parametrize(
        "objective, expected_id",
        [
            ("Analyse the stock price of AAPL over 6 months", "stock_analysis"),
            ("Generate a ticker report with OHLCV data", "stock_analysis"),
            ("Create a PowerPoint presentation for the Q3 review", "presentation"),
            ("Build a PPTX deck with 5 slides", "presentation"),
            ("Export the results to PDF", "pdf_report"),
            ("Render a PDF report from the markdown notes", "pdf_report"),
            ("Create a Word document summarising the findings", "docx_report"),
            ("Produce a DOCX report", "docx_report"),
            ("Transform the CSV file to filter by date", "csv_transform"),
            ("Aggregate and reshape the data.csv", "csv_transform"),
            ("Plot a bar chart of monthly sales", "chart"),
            ("Visualize the time series as a line graph", "chart"),
            ("Aggregate the research snippets into a brief", "research_brief"),
            ("Write a research summary from the sources", "research_brief"),
        ],
    )
    def test_objective_maps_to_recipe(self, objective, expected_id):
        result = recipe_for_objective(objective)
        assert result == expected_id, (
            f"Expected '{expected_id}' for objective '{objective}', got '{result}'"
        )

    def test_unrelated_objective_returns_none(self):
        result = recipe_for_objective(
            "Open the file explorer and navigate to Documents"
        )
        assert result is None


# ─────────────────────────────────────────────────────────────────────────── #
# Code generation — structural checks                                           #
# ─────────────────────────────────────────────────────────────────────────── #


class TestStockAnalysisCodeGen:
    def test_generates_nonempty_code(self):
        code = build_recipe_code("stock_analysis", {"ticker": "MSFT", "period": "3mo"})
        assert len(code.strip()) > 100

    def test_code_contains_ticker(self):
        code = build_recipe_code("stock_analysis", {"ticker": "GOOGL"})
        assert "GOOGL" in code

    def test_code_contains_result_markers(self):
        code = build_recipe_code("stock_analysis", {})
        assert "RESULT:" in code

    def test_code_is_valid_python_syntax(self):
        code = build_recipe_code("stock_analysis", {"ticker": "AAPL"})
        compile(code, "<stock_analysis>", "exec")  # raises SyntaxError if invalid

    def test_code_writes_to_sandbox_dir_env(self):
        code = build_recipe_code("stock_analysis", {})
        assert "AGENTOS_SANDBOX_DIR" in code


class TestPresentationCodeGen:
    def test_generates_nonempty_code(self):
        code = build_recipe_code(
            "presentation",
            {
                "title": "Q3 Results",
                "slides": [{"heading": "Summary", "bullets": ["Good quarter"]}],
            },
        )
        assert len(code.strip()) > 50

    def test_code_contains_title(self):
        code = build_recipe_code("presentation", {"title": "My Deck"})
        assert "My Deck" in code

    def test_code_is_valid_python(self):
        code = build_recipe_code("presentation", {})
        compile(code, "<presentation>", "exec")

    def test_code_contains_result_marker(self):
        code = build_recipe_code("presentation", {})
        assert "RESULT:" in code


class TestPdfReportCodeGen:
    def test_generates_nonempty_code(self):
        code = build_recipe_code("pdf_report", {"content": "# Hello\n\nWorld."})
        assert len(code.strip()) > 50

    def test_code_is_valid_python(self):
        code = build_recipe_code("pdf_report", {})
        compile(code, "<pdf_report>", "exec")

    def test_code_has_result_marker(self):
        code = build_recipe_code("pdf_report", {})
        assert "RESULT:" in code


class TestCsvTransformCodeGen:
    def test_generates_nonempty_code(self):
        code = build_recipe_code(
            "csv_transform",
            {
                "input_path": "data.csv",
                "output_file": "filtered.csv",
                "operations": [{"kind": "head", "n": 50}],
            },
        )
        assert len(code.strip()) > 50

    def test_code_is_valid_python(self):
        code = build_recipe_code("csv_transform", {})
        compile(code, "<csv_transform>", "exec")


class TestChartCodeGen:
    def test_generates_nonempty_code(self):
        code = build_recipe_code(
            "chart",
            {
                "chart_type": "bar",
                "x_data": ["Jan", "Feb", "Mar"],
                "y_data": [10, 20, 15],
                "title": "Monthly Sales",
            },
        )
        assert len(code.strip()) > 50

    def test_code_is_valid_python(self):
        code = build_recipe_code("chart", {})
        compile(code, "<chart>", "exec")

    def test_code_has_result_marker(self):
        code = build_recipe_code("chart", {})
        assert "RESULT:" in code


class TestResearchBriefCodeGen:
    def test_generates_nonempty_code(self):
        code = build_recipe_code(
            "research_brief",
            {
                "topic": "AI Safety",
                "snippets": [{"source": "arXiv", "text": "Recent work shows..."}],
            },
        )
        assert len(code.strip()) > 50

    def test_code_is_valid_python(self):
        code = build_recipe_code("research_brief", {})
        compile(code, "<research_brief>", "exec")


# ─────────────────────────────────────────────────────────────────────────── #
# Error handling                                                                #
# ─────────────────────────────────────────────────────────────────────────── #


def test_build_recipe_code_raises_for_unknown_recipe():
    with pytest.raises(KeyError):
        build_recipe_code("recipe_that_does_not_exist_xyz", {})
