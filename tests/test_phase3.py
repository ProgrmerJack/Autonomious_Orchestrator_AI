"""Tests for Phase 3 + Phase 1 additions.

Covers:
- ToolExecutor: sandboxed code execution, security validation, result parsing
- VLMAdapter:   ClassicalVLMAdapter scene understanding, element location
- HierarchicalTaskDecomposer: new analysis + tool_use patterns
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from agentos_orchestrator.cognition.tool_executor import (
    ToolExecutor,
    ToolResult,
    QuantAnalysisRequest,
    get_template,
    register_template,
)
from agentos_orchestrator.cognition.vlm_adapter import (
    ClassicalVLMAdapter,
    SceneUnderstanding,
    VLMElement,
    create_adapter,
)
from agentos_orchestrator.cognition.hierarchical_task_decomposer import (
    HierarchicalTaskDecomposer,
    TaskHierarchy,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _white_screenshot(size: tuple[int, int] = (640, 480)) -> Image.Image:
    img = Image.new("RGB", size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw a fake button
    draw.rectangle([100, 200, 220, 240], fill=(70, 130, 180))
    draw.text((120, 212), "Submit", fill="white")
    # Draw a fake input field
    draw.rectangle([100, 260, 400, 290], outline=(80, 80, 80), fill="white")
    return img


def _dark_screenshot(size: tuple[int, int] = (640, 480)) -> Image.Image:
    img = Image.new("RGB", size, color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 200, 220, 240], fill=(60, 60, 180))
    draw.text((120, 212), "Run", fill="white")
    return img


# ─────────────────────────────────────────────────────────────────────────── #
# ToolExecutor tests                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class ToolExecutorBasicTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_root=self._tmpdir)

    def test_simple_arithmetic_runs(self) -> None:
        req = QuantAnalysisRequest(
            objective="compute 2+2",
            code='print("RESULT: answer=4")\nprint(2 + 2)',
        )
        result = self.executor.run(req)
        self.assertTrue(result.success)
        self.assertEqual(result.parsed_results.get("answer"), 4)

    def test_stdout_captured(self) -> None:
        req = QuantAnalysisRequest(
            objective="test stdout",
            code='print("hello world")',
        )
        result = self.executor.run(req)
        self.assertIn("hello world", result.stdout)

    def test_elapsed_ms_positive(self) -> None:
        req = QuantAnalysisRequest(
            objective="timing test",
            code="x = sum(range(10000))",
        )
        result = self.executor.run(req)
        self.assertGreater(result.elapsed_ms, 0.0)

    def test_syntax_error_does_not_crash_executor(self) -> None:
        req = QuantAnalysisRequest(
            objective="broken code",
            code="def incomplete(",
        )
        result = self.executor.run(req)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error or result.stderr)

    def test_timeout_respected(self) -> None:
        req = QuantAnalysisRequest(
            objective="infinite loop",
            code="while True: pass",
            timeout_seconds=2,
        )
        result = self.executor.run(req)
        self.assertFalse(result.success)
        self.assertIn("timed out", (result.error or "").lower())

    def test_artefact_files_collected(self) -> None:
        req = QuantAnalysisRequest(
            objective="write a file",
            code=(
                'import os\n'
                'with open("output.txt", "w") as f:\n'
                '    f.write("artefact")\n'
            ),
        )
        result = self.executor.run(req)
        self.assertTrue(result.success)
        names = [a.name for a in result.artefacts]
        self.assertIn("output.txt", names)

    def test_numpy_available_in_sandbox(self) -> None:
        req = QuantAnalysisRequest(
            objective="numpy test",
            code=(
                "import numpy as np\n"
                "arr = np.array([1, 2, 3])\n"
                "print('RESULT: mean=' + str(arr.mean()))\n"
            ),
        )
        result = self.executor.run(req)
        self.assertTrue(result.success, msg=result.error or result.stderr)
        self.assertAlmostEqual(float(result.parsed_results["mean"]), 2.0)

    def test_pandas_available_in_sandbox(self) -> None:
        try:
            import importlib
            importlib.import_module("pandas")
        except ModuleNotFoundError:
            self.skipTest("pandas not installed in venv")
        req = QuantAnalysisRequest(
            objective="pandas test",
            code=(
                "import pandas as pd\n"
                "df = pd.DataFrame({'a': [1, 2, 3]})\n"
                "print('RESULT: rows=' + str(len(df)))\n"
            ),
        )
        result = self.executor.run(req)
        self.assertTrue(result.success, msg=result.error or result.stderr)
        self.assertEqual(result.parsed_results.get("rows"), 3)

    def test_result_summary_ok(self) -> None:
        req = QuantAnalysisRequest(
            objective="summary test",
            code='print("done")',
        )
        result = self.executor.run(req)
        summary = result.summary()
        self.assertTrue(summary.startswith("[TOOL_OK]"))

    def test_result_summary_error(self) -> None:
        req = QuantAnalysisRequest(
            objective="summary error test",
            code="raise ValueError('deliberate')",
        )
        result = self.executor.run(req)
        summary = result.summary()
        self.assertTrue(summary.startswith("[TOOL_ERROR]"))


class ToolExecutorSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_root=self._tmpdir)

    def test_blocked_os_system_call(self) -> None:
        req = QuantAnalysisRequest(
            objective="security check",
            code="import os; os.system('echo pwned')",
        )
        result = self.executor.run(req)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("os.system", result.error)

    def test_blocked_subprocess_import(self) -> None:
        req = QuantAnalysisRequest(
            objective="security check",
            code="import subprocess; subprocess.run(['echo', 'hi'])",
        )
        result = self.executor.run(req)
        self.assertFalse(result.success)
        self.assertIn("subprocess", result.error or "")

    def test_blocked_eval(self) -> None:
        req = QuantAnalysisRequest(
            objective="eval check",
            code='result = eval("1+1")',
        )
        result = self.executor.run(req)
        self.assertFalse(result.success)
        self.assertIn("eval(", result.error or "")

    def test_write_outside_sandbox_blocked(self) -> None:
        # Try to write to the workspace root (outside the run_dir)
        import os
        parent = str(Path(self._tmpdir).parent).replace("\\", "/")
        req = QuantAnalysisRequest(
            objective="path escape",
            code=f'with open("{parent}/evil.txt", "w") as f: f.write("bad")',
        )
        result = self.executor.run(req)
        self.assertFalse(result.success)

    def test_vetted_package_allow_list(self) -> None:
        req = QuantAnalysisRequest(
            objective="package check",
            code="import numpy",
            allowed_packages=["evil_package"],
        )
        result = self.executor.run(req)
        self.assertFalse(result.success)
        self.assertIn("allow-list", result.error or "")


class ToolExecutorTemplateTests(unittest.TestCase):
    def test_template_registry_contains_defaults(self) -> None:
        for name in ("portfolio_stats", "rolling_volatility", "market_regime_hmm"):
            tmpl = get_template(name)
            self.assertIsNotNone(tmpl, f"Template '{name}' missing")
            self.assertIn("RESULT:", tmpl)

    def test_custom_template_registration(self) -> None:
        register_template("custom_test", 'print("RESULT: custom=1")')
        tmpl = get_template("custom_test")
        self.assertIsNotNone(tmpl)

    def test_build_quant_analysis_code_generates_valid_python(self) -> None:
        tmpdir = tempfile.mkdtemp()
        executor = ToolExecutor(workspace_root=tmpdir)
        code = executor.build_quant_analysis_code(
            "compute vol", tickers=["SPY"], period="1mo"
        )
        self.assertIsInstance(code, str)
        self.assertIn("import", code)
        self.assertIn("RESULT:", code)
        # It should compile without syntax errors
        compile(code, "<generated>", "exec")


# ─────────────────────────────────────────────────────────────────────────── #
# VLMAdapter tests                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class ClassicalVLMAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = ClassicalVLMAdapter()

    def test_understand_scene_returns_scene_understanding(self) -> None:
        screenshot = _white_screenshot()
        scene = self.adapter.understand_scene(screenshot, "click submit")
        self.assertIsInstance(scene, SceneUnderstanding)
        self.assertIsInstance(scene.description, str)
        self.assertGreater(len(scene.description), 0)

    def test_understand_scene_dark_mode_detects_dark_theme(self) -> None:
        screenshot = _dark_screenshot()
        scene = self.adapter.understand_scene(screenshot)
        self.assertEqual(scene.theme, "dark")

    def test_understand_scene_light_mode_detects_light_theme(self) -> None:
        screenshot = _white_screenshot()
        scene = self.adapter.understand_scene(screenshot)
        self.assertEqual(scene.theme, "light")

    def test_understand_scene_has_latency_ms(self) -> None:
        scene = self.adapter.understand_scene(_white_screenshot())
        self.assertGreater(scene.latency_ms, 0.0)

    def test_understand_scene_adapter_name(self) -> None:
        scene = self.adapter.understand_scene(_white_screenshot())
        self.assertIn("classical", scene.adapter_name)

    def test_locate_elements_returns_list(self) -> None:
        screenshot = _white_screenshot()
        elements = self.adapter.locate_elements(screenshot, "submit button")
        self.assertIsInstance(elements, list)

    def test_locate_elements_sorted_by_confidence(self) -> None:
        screenshot = _white_screenshot()
        elements = self.adapter.locate_elements(screenshot, "button")
        if len(elements) >= 2:
            self.assertGreaterEqual(elements[0].confidence, elements[1].confidence)

    def test_extract_text_returns_string(self) -> None:
        img = Image.new("RGB", (200, 50), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Hello", fill="black")
        text = self.adapter.extract_text(img)
        self.assertIsInstance(text, str)

    def test_extract_text_with_region(self) -> None:
        screenshot = _white_screenshot()
        text = self.adapter.extract_text(screenshot, region=(100, 200, 220, 240))
        self.assertIsInstance(text, str)

    def test_create_adapter_factory_classical(self) -> None:
        adapter = create_adapter("classical")
        self.assertIsInstance(adapter, ClassicalVLMAdapter)

    def test_create_adapter_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_adapter("nonexistent_model_xyz")

    def test_vlm_element_cx_cy(self) -> None:
        elem = VLMElement(x=100, y=200, width=60, height=30, label="OK", element_type="button")
        self.assertEqual(elem.cx, 130)
        self.assertEqual(elem.cy, 215)

    def test_scene_understanding_interactive_elements(self) -> None:
        scene = self.adapter.understand_scene(_white_screenshot(), "click button")
        interactive = scene.interactive_elements
        self.assertIsInstance(interactive, list)
        for e in interactive:
            self.assertIn(e.element_type, {"button", "text_field", "checkbox", "dropdown", "slider"})


# ─────────────────────────────────────────────────────────────────────────── #
# Decomposer: new patterns                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class DecomposerPhase3PatternTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decomposer = HierarchicalTaskDecomposer()

    def test_analyse_stock_triggers_analysis_hierarchy(self) -> None:
        h = self.decomposer.decompose("analyse the stock market for AAPL")
        names = [opt.name for opt in h.execution_sequence]
        self.assertIn("collect_data_via_tool", names)
        self.assertIn("run_analysis_code", names)

    def test_quantitative_analysis_triggers_analysis_hierarchy(self) -> None:
        h = self.decomposer.decompose("compute quantitative analysis of portfolio volatility")
        names = [opt.name for opt in h.execution_sequence]
        self.assertIn("collect_data_via_tool", names)

    def test_portfolio_analysis_triggers_analysis_hierarchy(self) -> None:
        h = self.decomposer.decompose("build a portfolio analysis with Sharpe ratios")
        names = [opt.name for opt in h.execution_sequence]
        self.assertIn("run_analysis_code", names)

    def test_run_script_triggers_tool_use_hierarchy(self) -> None:
        h = self.decomposer.decompose("run a Python script to batch rename files")
        names = [opt.name for opt in h.execution_sequence]
        self.assertIn("plan_script", names)
        self.assertIn("execute_script", names)

    def test_execute_command_triggers_tool_use_hierarchy(self) -> None:
        h = self.decomposer.decompose("execute the data pipeline command")
        names = [opt.name for opt in h.execution_sequence]
        self.assertIn("execute_script", names)

    def test_analysis_hierarchy_has_three_options(self) -> None:
        h = self.decomposer.decompose("financial data analysis of MSFT")
        # collect, analyse, report
        self.assertEqual(len(h.execution_sequence), 3)

    def test_tool_use_hierarchy_has_three_options(self) -> None:
        h = self.decomposer.decompose("run a compute script")
        self.assertEqual(len(h.execution_sequence), 3)

    def test_analysis_completion_probability_reasonable(self) -> None:
        h = self.decomposer.decompose("analyse stock price data")
        p = self.decomposer.estimate_completion_probability(h)
        self.assertGreater(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_option_library_contains_analysis_options(self) -> None:
        library_names = list(self.decomposer._option_library.keys())
        self.assertIn("collect_data_via_tool", library_names)
        self.assertIn("plan_script", library_names)

    def test_research_still_works(self) -> None:
        h = self.decomposer.decompose("research the best CRM tools")
        names = [opt.name for opt in h.execution_sequence]
        self.assertIn("gather_information", names)

    def test_figma_falls_back_to_content_hierarchy(self) -> None:
        h = self.decomposer.decompose("create a wireframe design in Figma")
        names = [opt.name for opt in h.execution_sequence]
        # Should match "create" → content hierarchy
        self.assertIn("create_content", names)


if __name__ == "__main__":
    unittest.main()
