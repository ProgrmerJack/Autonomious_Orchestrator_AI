"""Hierarchical Task Decomposer — Option-Based Planning for Long-Horizon Goals.

Addresses "The Definition of Any Task" by breaking ambiguous objectives into
concrete, tractable subtasks using the Options framework (Sutton et al.).

Key insight: Instead of one massive MCTS tree for "research CRMs, sign up,
map contacts", we use a hierarchy:

  Macro: ResearchAndSelectCRM
    ├─ Option: GatherCRMInfo (browser search)
    ├─ Option: CompareFeatures (spreadsheet analysis)
    └─ Option: SelectBest (decision)

  Macro: SignUpForTrial
    ├─ Option: NavigateToSignup (browser navigation)
    ├─ Option: FillForm (form completion)
    └─ Option: ConfirmAccount (email verification)

Each option has:
- Initiation condition (when can it start?)
- Policy (sequence of actions or sub-options)
- Termination condition (when is it done?)
- Expected reward / success probability

This reduces the effective branching factor at each level from
hundreds of UI actions to ~5-10 options.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from agentos_orchestrator.os_control.base import UiAction

from .abstract_world_model import AbstractUIState
from .semantic_memory import SemanticEpisodicMemory


@dataclass
class Option:
    """A temporally extended action (macro-action).

    Options framework: initiation → policy → termination.
    """

    name: str
    description: str
    # Can this option start given current state?
    initiation_check: Callable[[AbstractUIState], bool]
    # What to execute (list of primitive actions or sub-option names)
    policy: list[str | UiAction]
    # Is this option finished?
    termination_check: Callable[[AbstractUIState], bool]
    # Estimated success probability [0, 1]
    success_probability: float = 0.5
    # Expected steps to completion
    expected_duration: int = 5
    # Required preconditions
    preconditions: list[str] = field(default_factory=list)
    # Postconditions (what should be true after)
    postconditions: list[str] = field(default_factory=list)
    # Sub-options (for hierarchical decomposition)
    sub_options: list["Option"] = field(default_factory=list)
    # Has this option been executed successfully before?
    execution_count: int = 0
    success_count: int = 0

    def can_start(self, state: AbstractUIState) -> bool:
        return self.initiation_check(state)

    def is_done(self, state: AbstractUIState) -> bool:
        return self.termination_check(state)

    @property
    def empirical_success_rate(self) -> float:
        return self.success_count / max(self.execution_count, 1)


@dataclass
class TaskHierarchy:
    """A decomposed task as a tree of options."""

    root_objective: str
    top_level_options: list[Option] = field(default_factory=list)
    # Flattened sequence for execution
    execution_sequence: list[Option] = field(default_factory=list)
    # Current position in execution
    current_index: int = 0

    def next_option(self) -> Option | None:
        if self.current_index < len(self.execution_sequence):
            opt = self.execution_sequence[self.current_index]
            self.current_index += 1
            return opt
        return None

    def has_more(self) -> bool:
        return self.current_index < len(self.execution_sequence)

    def mark_current_success(self) -> None:
        idx = self.current_index - 1
        if 0 <= idx < len(self.execution_sequence):
            self.execution_sequence[idx].execution_count += 1
            self.execution_sequence[idx].success_count += 1

    def mark_current_failure(self) -> None:
        idx = self.current_index - 1
        if 0 <= idx < len(self.execution_sequence):
            self.execution_sequence[idx].execution_count += 1


class HierarchicalTaskDecomposer:
    """Decomposes arbitrary objectives into option hierarchies.

    Uses a library of reusable options + semantic matching to select
    the right decomposition pattern for each objective.
    """

    def __init__(self, memory: SemanticEpisodicMemory | None = None) -> None:
        self.memory = memory
        self._option_library: dict[str, Option] = {}
        self._build_option_library()

    def decompose(
        self, objective: str, current_state: AbstractUIState | None = None
    ) -> TaskHierarchy:
        """Decompose an objective into a hierarchy of executable options.

        Strategy:
        1. Match objective against known patterns
        2. Retrieve similar past decompositions from memory
        3. Build option tree with initiation/termination conditions
        4. Flatten to execution sequence
        """
        lower = objective.lower()

        if self._should_bootstrap_unknown_surface(current_state, lower):
            return self._build_unknown_surface_hierarchy(objective)

        # Try pattern-based decomposition
        hierarchy = self._pattern_decompose(objective, lower, current_state)
        if hierarchy is not None:
            return hierarchy

        # Fallback: retrieve from episodic memory
        if self.memory is not None:
            similar = self.memory.retrieve_similar(objective, top_k=3)
            for event in similar:
                if "hierarchy" in event:
                    # Reuse past decomposition
                    return self._rebuild_hierarchy(event["hierarchy"], objective)

        # Ultimate fallback: single exploratory option
        return self._exploratory_hierarchy(objective)

    def replan_on_failure(
        self,
        hierarchy: TaskHierarchy,
        failed_option: Option,
        state: AbstractUIState,
    ) -> TaskHierarchy:
        """When an option fails, try alternative sub-options or fall back."""
        # Find alternatives for the failed option
        alternatives = self._find_alternatives(failed_option)
        if alternatives:
            # Insert alternative after current position
            idx = hierarchy.current_index
            hierarchy.execution_sequence = (
                hierarchy.execution_sequence[:idx]
                + alternatives
                + hierarchy.execution_sequence[idx:]
            )
        else:
            # No alternatives — decompose further
            sub_decomp = self._decompose_option(failed_option, state)
            if sub_decomp:
                idx = hierarchy.current_index
                hierarchy.execution_sequence = (
                    hierarchy.execution_sequence[:idx]
                    + sub_decomp.execution_sequence
                    + hierarchy.execution_sequence[idx:]
                )
        return hierarchy

    def estimate_completion_probability(self, hierarchy: TaskHierarchy) -> float:
        """Estimate P(success) for the entire task hierarchy."""
        if not hierarchy.execution_sequence:
            return 0.0
        probs = [opt.success_probability for opt in hierarchy.execution_sequence]
        # Assume some independence between options
        return float(np.prod(probs) ** 0.5)  # Geometric mean softening

    # ------------------------------------------------------------------ #
    # Pattern-Based Decomposition
    # ------------------------------------------------------------------ #

    def _pattern_decompose(
        self,
        objective: str,
        lower: str,
        state: AbstractUIState | None,
    ) -> TaskHierarchy | None:
        """Match objective against known high-level patterns."""

        # Pattern: Research + Compare + Select
        if any(
            kw in lower
            for kw in {"research", "compare", "find the best", "evaluate", "choose"}
        ):
            return self._build_research_hierarchy(objective)

        # Pattern: Sign Up / Register
        if any(kw in lower for kw in {"sign up", "register", "create account", "join"}):
            return self._build_signup_hierarchy(objective)

        # Pattern: Fill Form / Input Data
        if any(
            kw in lower for kw in {"fill", "input", "enter", "map", "import", "upload"}
        ):
            return self._build_form_hierarchy(objective)

        # Pattern: Open + Use + Close
        if any(kw in lower for kw in {"open", "launch", "start"}) and any(
            kw in lower for kw in {"use", "with", "and then"}
        ):
            return self._build_open_use_hierarchy(objective)

        # Pattern: Tool Use / Script Execution  (checked BEFORE file_op so that
        # "run a batch rename script" or "execute pipeline" wins over file-op)
        if any(
            kw in lower
            for kw in {
                "run",
                "execute",
                "script",
                "code",
                "terminal",
                "command",
                "automate",
                "batch",
                "pipeline",
                "compute",
                "calculate",
            }
        ) and not any(
            # Don't intercept when the context is clearly data/quant-only
            kw in lower
            for kw in {"stock", "market", "portfolio", "financial", "trading"}
        ):
            return self._build_tool_use_hierarchy(objective)

        # Pattern: File Operations
        if any(
            kw in lower
            for kw in {"save", "export", "download", "move", "copy", "delete", "rename"}
        ):
            return self._build_file_op_hierarchy(objective)

        # Pattern: Content Creation
        if any(
            kw in lower
            for kw in {"write", "create", "draw", "make", "generate", "compose"}
        ):
            return self._build_content_hierarchy(objective)

        # Pattern: Search + Extract
        if any(kw in lower for kw in {"search", "find", "look up", "query"}) and any(
            kw in lower for kw in {"extract", "get", "retrieve", "copy"}
        ):
            return self._build_search_extract_hierarchy(objective)

        # Pattern: Quantitative / Data Analysis
        # This is THE most important new pattern: recognising when the agent should
        # write code rather than click a UI.
        if any(
            kw in lower
            for kw in {
                "analyse",
                "analyze",
                "analysis",
                "stock",
                "market",
                "quant",
                "quantitative",
                "data",
                "statistics",
                "forecast",
                "volatility",
                "portfolio",
                "trading",
                "price",
                "chart",
                "regression",
                "correlation",
                "backtest",
                "var",
                "sharpe",
                "financial",
            }
        ):
            return self._build_analysis_hierarchy(objective)

        return None

    def _build_research_hierarchy(self, objective: str) -> TaskHierarchy:
        """Decompose research objectives: Research X, compare, select best."""
        gather = Option(
            name="gather_information",
            description=f"Search for and collect information about: {objective}",
            initiation_check=lambda s: s.app_context == "browser",
            policy=[
                "open_search",
                "enter_query",
                "scan_results",
                "open_relevant_pages",
            ],
            termination_check=lambda s: (
                len(s.elements) > 3
                and any(
                    e.element_type in {"table", "panel", "text_block"}
                    for e in s.elements
                )
            ),
            success_probability=0.7,
            expected_duration=8,
            postconditions=["Information has been gathered and is visible"],
        )
        compare = Option(
            name="compare_and_analyze",
            description="Compare gathered information and identify key differences",
            initiation_check=lambda s: any(
                e.element_type in {"table", "text_block"} for e in s.elements
            ),
            policy=["extract_data", "create_comparison_table", "highlight_differences"],
            termination_check=lambda s: any(
                e.semantic_label in {"comparison", "analysis", "summary"}
                for e in s.elements
            ),
            success_probability=0.6,
            expected_duration=6,
            preconditions=["Information has been gathered"],
            postconditions=["Comparison is visible and analyzable"],
        )
        select = Option(
            name="make_selection",
            description="Select the best option based on analysis",
            initiation_check=lambda s: any(
                e.semantic_label in {"comparison", "analysis"} for e in s.elements
            ),
            policy=["review_comparison", "click_best_option", "confirm_selection"],
            termination_check=lambda s: s.task_progress.get("selection_made", 0) > 0.8,
            success_probability=0.8,
            expected_duration=3,
            preconditions=["Comparison has been completed"],
            postconditions=["Best option has been selected"],
        )
        return self._build_hierarchy(objective, [gather, compare, select])

    def _build_signup_hierarchy(self, objective: str) -> TaskHierarchy:
        """Decompose sign-up objectives: Navigate, fill form, confirm."""
        navigate = Option(
            name="navigate_to_signup",
            description="Navigate to the sign-up or registration page",
            initiation_check=lambda s: s.app_context == "browser",
            policy=["find_signup_link", "click_signup", "wait_for_form"],
            termination_check=lambda s: any(
                e.element_type == "text_field" and "email" in e.semantic_label.lower()
                for e in s.elements
            ),
            success_probability=0.75,
            expected_duration=4,
        )
        fill_form = Option(
            name="fill_registration_form",
            description="Complete all required fields in the registration form",
            initiation_check=lambda s: any(
                e.element_type == "text_field" for e in s.elements
            ),
            policy=[
                "fill_email",
                "fill_password",
                "fill_name",
                "check_terms",
                "click_submit",
            ],
            termination_check=lambda s: s.task_progress.get("form_submitted", 0) > 0.9,
            success_probability=0.6,
            expected_duration=7,
            preconditions=["Sign-up form is visible"],
        )
        confirm = Option(
            name="confirm_account",
            description="Verify account creation via confirmation step",
            initiation_check=lambda s: s.task_progress.get("form_submitted", 0) > 0.5,
            policy=["check_email", "click_confirmation_link", "verify_login"],
            termination_check=lambda s: (
                s.task_progress.get("account_confirmed", 0) > 0.9
            ),
            success_probability=0.5,
            expected_duration=10,
            preconditions=["Form has been submitted"],
        )
        return self._build_hierarchy(objective, [navigate, fill_form, confirm])

    def _build_form_hierarchy(self, objective: str) -> TaskHierarchy:
        """Decompose form-filling objectives."""
        locate = Option(
            name="locate_form",
            description="Find and focus the correct input form",
            initiation_check=lambda s: len(s.elements) > 0,
            policy=["scan_for_form", "focus_form"],
            termination_check=lambda s: (
                s.focus_region in {"main", "modal"}
                and any(e.element_type == "text_field" for e in s.elements)
            ),
            success_probability=0.8,
            expected_duration=3,
        )
        fill = Option(
            name="fill_fields",
            description="Enter required data into form fields",
            initiation_check=lambda s: any(
                e.element_type == "text_field" for e in s.elements
            ),
            policy=["fill_field_1", "fill_field_2", "fill_field_3", "verify_entries"],
            termination_check=lambda s: s.task_progress.get("form_complete", 0) > 0.9,
            success_probability=0.7,
            expected_duration=6,
            preconditions=["Form is focused and visible"],
        )
        submit = Option(
            name="submit_form",
            description="Submit the completed form",
            initiation_check=lambda s: s.task_progress.get("form_complete", 0) > 0.8,
            policy=["click_submit", "wait_for_response", "verify_success"],
            termination_check=lambda s: s.task_progress.get("form_submitted", 0) > 0.9,
            success_probability=0.75,
            expected_duration=3,
            preconditions=["Form is complete"],
        )
        return self._build_hierarchy(objective, [locate, fill, submit])

    def _build_open_use_hierarchy(self, objective: str) -> TaskHierarchy:
        """Decompose open-and-use objectives."""
        launch = Option(
            name="launch_application",
            description="Open the target application",
            initiation_check=lambda s: True,
            policy=["find_app", "launch_app", "wait_for_window"],
            termination_check=lambda s: (
                s.app_context != "unknown" and len(s.elements) > 2
            ),
            success_probability=0.85,
            expected_duration=5,
        )
        use = Option(
            name="use_application",
            description=f"Use the application to accomplish: {objective}",
            initiation_check=lambda s: s.app_context != "unknown",
            policy=["interact_with_app"],  # Will be expanded by micro-executor
            termination_check=lambda s: s.task_progress.get("task_complete", 0) > 0.8,
            success_probability=0.5,
            expected_duration=15,
            preconditions=["Application is open"],
        )
        cleanup = Option(
            name="cleanup_and_save",
            description="Save work and close application if needed",
            initiation_check=lambda s: s.task_progress.get("task_complete", 0) > 0.5,
            policy=["save_work", "confirm_save", "close_app"],
            termination_check=lambda s: (
                not s.active_modal and s.task_progress.get("saved", 0) > 0.9
            ),
            success_probability=0.8,
            expected_duration=4,
            preconditions=["Task is substantially complete"],
        )
        return self._build_hierarchy(objective, [launch, use, cleanup])

    def _build_file_op_hierarchy(self, objective: str) -> TaskHierarchy:
        """Decompose file operation objectives."""
        lower = objective.lower()
        if "move" in lower or "copy" in lower:
            steps = [
                Option(
                    name="select_source",
                    description="Select the source file(s)",
                    initiation_check=lambda s: (
                        s.app_context in {"file_explorer", "browser"}
                    ),
                    policy=["navigate_to_source", "select_file"],
                    termination_check=lambda s: (
                        s.task_progress.get("source_selected", 0) > 0.9
                    ),
                    success_probability=0.8,
                    expected_duration=4,
                ),
                Option(
                    name="perform_operation",
                    description="Execute the file operation",
                    initiation_check=lambda s: (
                        s.task_progress.get("source_selected", 0) > 0.8
                    ),
                    policy=["copy_or_move", "navigate_to_dest", "paste_or_drop"],
                    termination_check=lambda s: (
                        s.task_progress.get("operation_done", 0) > 0.9
                    ),
                    success_probability=0.75,
                    expected_duration=5,
                ),
                Option(
                    name="verify_result",
                    description="Confirm the file exists at destination",
                    initiation_check=lambda s: (
                        s.task_progress.get("operation_done", 0) > 0.5
                    ),
                    policy=["check_destination", "verify_file"],
                    termination_check=lambda s: (
                        s.task_progress.get("verified", 0) > 0.9
                    ),
                    success_probability=0.9,
                    expected_duration=3,
                ),
            ]
            return self._build_hierarchy(objective, steps)

        # Generic save/export
        return self._build_hierarchy(
            objective,
            [
                Option(
                    name="initiate_save",
                    description="Open save/export dialog",
                    initiation_check=lambda s: True,
                    policy=["trigger_save", "wait_for_dialog"],
                    termination_check=lambda s: s.layout_mode == "modal_open",
                    success_probability=0.85,
                    expected_duration=3,
                ),
                Option(
                    name="configure_and_confirm",
                    description="Set filename/location and confirm",
                    initiation_check=lambda s: s.layout_mode == "modal_open",
                    policy=["set_filename", "choose_location", "click_save"],
                    termination_check=lambda s: s.task_progress.get("saved", 0) > 0.9,
                    success_probability=0.8,
                    expected_duration=4,
                ),
            ],
        )

    def _build_content_hierarchy(self, objective: str) -> TaskHierarchy:
        """Decompose content creation objectives."""
        return self._build_hierarchy(
            objective,
            [
                Option(
                    name="setup_workspace",
                    description="Open editor and prepare workspace",
                    initiation_check=lambda s: True,
                    policy=["open_editor", "new_document", "focus_workspace"],
                    termination_check=lambda s: (
                        s.app_context in {"text_editor", "media", "other"}
                        and s.focus_region == "main"
                    ),
                    success_probability=0.8,
                    expected_duration=4,
                ),
                Option(
                    name="create_content",
                    description=f"Create the required content for: {objective}",
                    initiation_check=lambda s: s.focus_region == "main",
                    policy=["type_content", "insert_elements", "format_content"],
                    termination_check=lambda s: (
                        s.task_progress.get("content_created", 0) > 0.85
                    ),
                    success_probability=0.5,
                    expected_duration=12,
                ),
                Option(
                    name="finalize_content",
                    description="Review, save, and export if needed",
                    initiation_check=lambda s: (
                        s.task_progress.get("content_created", 0) > 0.7
                    ),
                    policy=["review_content", "save_document", "export_if_needed"],
                    termination_check=lambda s: s.task_progress.get("saved", 0) > 0.9,
                    success_probability=0.75,
                    expected_duration=5,
                ),
            ],
        )

    def _build_search_extract_hierarchy(self, objective: str) -> TaskHierarchy:
        """Decompose search-and-extract objectives."""
        return self._build_hierarchy(
            objective,
            [
                Option(
                    name="execute_search",
                    description="Perform the search query",
                    initiation_check=lambda s: (
                        s.app_context in {"browser", "file_explorer", "other"}
                    ),
                    policy=["focus_search", "enter_query", "execute_search"],
                    termination_check=lambda s: any(
                        e.element_type in {"table", "panel", "text_block", "link"}
                        for e in s.elements
                    ),
                    success_probability=0.8,
                    expected_duration=4,
                ),
                Option(
                    name="extract_relevant_data",
                    description="Identify and extract the target information",
                    initiation_check=lambda s: any(
                        e.element_type in {"table", "panel", "text_block"}
                        for e in s.elements
                    ),
                    policy=["scan_results", "select_relevant", "copy_or_extract"],
                    termination_check=lambda s: (
                        s.task_progress.get("data_extracted", 0) > 0.9
                    ),
                    success_probability=0.6,
                    expected_duration=6,
                ),
                Option(
                    name="organize_output",
                    description="Place extracted data in the desired format/location",
                    initiation_check=lambda s: (
                        s.task_progress.get("data_extracted", 0) > 0.5
                    ),
                    policy=["open_destination", "paste_data", "format_output"],
                    termination_check=lambda s: (
                        s.task_progress.get("output_ready", 0) > 0.9
                    ),
                    success_probability=0.7,
                    expected_duration=5,
                ),
            ],
        )

    def _build_analysis_hierarchy(self, objective: str) -> TaskHierarchy:
        """Decompose data / quantitative analysis objectives into tool-use steps.

        Key insight: instead of clicking Yahoo Finance, the agent generates and
        runs Python code in the ToolExecutor sandbox.  This is ~100x faster and
        more reliable than GUI navigation for data-heavy tasks.
        """
        gather = Option(
            name="collect_data_via_tool",
            description=f"Write and run Python data-collection code for: {objective}",
            initiation_check=lambda s: True,
            policy=[
                "generate_data_fetch_code",
                "execute_in_sandbox",
                "verify_data_shape",
            ],
            termination_check=lambda s: s.task_progress.get("data_ready", 0) > 0.9,
            success_probability=0.75,
            expected_duration=5,
            postconditions=["Raw data is available in sandbox memory"],
        )
        analyse = Option(
            name="run_analysis_code",
            description=f"Perform quantitative analysis: {objective}",
            initiation_check=lambda s: s.task_progress.get("data_ready", 0) > 0.5,
            policy=[
                "generate_analysis_code",
                "execute_in_sandbox",
                "parse_result_lines",
            ],
            termination_check=lambda s: (
                s.task_progress.get("analysis_complete", 0) > 0.9
            ),
            success_probability=0.7,
            expected_duration=8,
            preconditions=["Data is available"],
            postconditions=["Analysis results are in structured form"],
        )
        report = Option(
            name="format_and_present_results",
            description="Format analysis output for display or export",
            initiation_check=lambda s: (
                s.task_progress.get("analysis_complete", 0) > 0.7
            ),
            policy=[
                "render_summary",
                "open_output_target",
                "paste_or_write_results",
            ],
            termination_check=lambda s: s.task_progress.get("reported", 0) > 0.9,
            success_probability=0.8,
            expected_duration=4,
            preconditions=["Analysis is complete"],
        )
        return self._build_hierarchy(objective, [gather, analyse, report])

    def _build_tool_use_hierarchy(self, objective: str) -> TaskHierarchy:
        """Decompose explicit tool-use / script-execution objectives."""
        plan = Option(
            name="plan_script",
            description=f"Determine what code to write for: {objective}",
            initiation_check=lambda s: True,
            policy=["identify_libraries", "outline_logic", "draft_code"],
            termination_check=lambda s: s.task_progress.get("code_ready", 0) > 0.9,
            success_probability=0.8,
            expected_duration=4,
        )
        execute = Option(
            name="execute_script",
            description="Run the generated script in the sandboxed executor",
            initiation_check=lambda s: s.task_progress.get("code_ready", 0) > 0.7,
            policy=["send_to_tool_executor", "monitor_output", "handle_errors"],
            termination_check=lambda s: s.task_progress.get("script_complete", 0) > 0.9,
            success_probability=0.7,
            expected_duration=10,
            preconditions=["Code has been drafted"],
        )
        verify = Option(
            name="verify_script_output",
            description="Check script result is correct and use it",
            initiation_check=lambda s: s.task_progress.get("script_complete", 0) > 0.5,
            policy=["parse_stdout", "validate_result", "store_artefacts"],
            termination_check=lambda s: s.task_progress.get("verified", 0) > 0.9,
            success_probability=0.85,
            expected_duration=3,
        )
        return self._build_hierarchy(objective, [plan, execute, verify])

    def _exploratory_hierarchy(self, objective: str) -> TaskHierarchy:
        """Fallback: purely exploratory decomposition."""
        return self._build_hierarchy(
            objective,
            [
                Option(
                    name="explore_ui",
                    description=f"Explore the UI to understand how to: {objective}",
                    initiation_check=lambda s: True,
                    policy=["scan_elements", "try_interactions", "observe_changes"],
                    termination_check=lambda s: (
                        s.task_progress.get("understood", 0) > 0.7
                    ),
                    success_probability=0.3,
                    expected_duration=10,
                ),
                Option(
                    name="attempt_objective",
                    description=f"Try to accomplish: {objective}",
                    initiation_check=lambda s: (
                        s.task_progress.get("understood", 0) > 0.3
                    ),
                    policy=["execute_best_guess", "monitor_result"],
                    termination_check=lambda s: (
                        s.task_progress.get("task_complete", 0) > 0.7
                    ),
                    success_probability=0.2,
                    expected_duration=8,
                ),
            ],
        )

    def _build_unknown_surface_hierarchy(self, objective: str) -> TaskHierarchy:
        """Bootstrap unknown applications before task-specific execution."""
        return self._build_hierarchy(
            objective,
            [
                Option(
                    name="orient_surface",
                    description=(
                        "Infer the surface purpose, visible landmarks, and the safest "
                        f"control channels for: {objective}"
                    ),
                    initiation_check=lambda s: True,
                    policy=[
                        "inspect_window_chrome",
                        "infer_app_purpose",
                        "identify_primary_workspace",
                    ],
                    termination_check=lambda s: (
                        s.app_context != "unknown"
                        or self._interactive_count(s) >= 2
                        or s.focus_region in {"main", "modal"}
                    ),
                    success_probability=0.55,
                    expected_duration=4,
                ),
                Option(
                    name="discover_affordances",
                    description=(
                        "Read the UI, gather documentation or API hints, and test safe "
                        f"primitive controls for: {objective}"
                    ),
                    initiation_check=lambda s: True,
                    policy=[
                        "gather_docs_or_api_hints",
                        "probe_safe_primitives",
                        "record_affordance_effects",
                    ],
                    termination_check=lambda s: (
                        self._interactive_count(s) >= 2
                        or s.task_progress.get("understood", 0) > 0.4
                    ),
                    success_probability=0.45,
                    expected_duration=6,
                    preconditions=["The surface has been oriented"],
                ),
                Option(
                    name="attempt_grounded_objective",
                    description=(
                        "Use the best grounded control path to attempt the real objective: "
                        f"{objective}"
                    ),
                    initiation_check=lambda s: (
                        self._interactive_count(s) > 0
                        or s.task_progress.get("understood", 0) > 0.2
                    ),
                    policy=[
                        "select_best_affordance",
                        "execute_grounded_primitive",
                        "verify_outcome",
                    ],
                    termination_check=lambda s: (
                        s.task_progress.get("task_complete", 0) > 0.7
                        or s.task_progress.get("grounded_attempt", 0) > 0.6
                    ),
                    success_probability=0.35,
                    expected_duration=8,
                    preconditions=["At least one control path has been grounded"],
                ),
            ],
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_hierarchy(self, objective: str, options: list[Option]) -> TaskHierarchy:
        """Build TaskHierarchy from top-level options, flattening sub-options."""
        sequence: list[Option] = []
        for opt in options:
            sequence.extend(self._flatten_option(opt))
        return TaskHierarchy(
            root_objective=objective,
            top_level_options=options,
            execution_sequence=sequence,
            current_index=0,
        )

    def _flatten_option(self, option: Option) -> list[Option]:
        """Depth-first flatten of option + sub-options."""
        result: list[Option] = [option]
        for sub in option.sub_options:
            result.extend(self._flatten_option(sub))
        return result

    def _find_alternatives(self, failed_option: Option) -> list[Option]:
        """Find alternative options with similar goals."""
        alternatives: list[Option] = []
        for opt in self._option_library.values():
            if opt.name != failed_option.name:
                # Semantic similarity check
                if any(
                    kw in opt.description.lower()
                    for kw in failed_option.description.lower().split()[:3]
                ):
                    alternatives.append(opt)
        return alternatives[:3]  # Max 3 alternatives

    def _decompose_option(
        self, option: Option, state: AbstractUIState
    ) -> TaskHierarchy | None:
        """Try to decompose a single option further."""
        # Check if this option type has a known decomposition pattern
        for pattern_name, builder in self._decomposition_patterns().items():
            if pattern_name in option.name.lower():
                return builder(option.description)
        return None

    def _decomposition_patterns(self) -> dict[str, Callable[[str], TaskHierarchy]]:
        """Map option names to decomposition builders."""
        return {
            "research": self._build_research_hierarchy,
            "signup": self._build_signup_hierarchy,
            "form": self._build_form_hierarchy,
            "content": self._build_content_hierarchy,
        }

    def _rebuild_hierarchy(
        self, saved: dict[str, Any], objective: str
    ) -> TaskHierarchy:
        """Rebuild hierarchy from saved memory structure."""
        # Simplified: create from objective as if new
        return self.decompose(objective)

    @staticmethod
    def _interactive_count(state: AbstractUIState | None) -> int:
        if state is None:
            return 0
        return sum(1 for element in state.elements if element.is_interactive)

    def _should_bootstrap_unknown_surface(
        self,
        state: AbstractUIState | None,
        lower: str,
    ) -> bool:
        if state is None:
            return False
        if state.app_context not in {"unknown", "other"}:
            return False
        if any(
            kw in lower
            for kw in {
                "script",
                "code",
                "terminal",
                "command",
                "quant",
                "analysis",
                "stock",
                "market",
                "financial",
                "backtest",
            }
        ):
            return False
        return len(state.elements) < 4 or self._interactive_count(state) < 2

    def _build_option_library(self) -> None:
        """Pre-populate library of reusable options."""
        common_options = [
            self._build_research_hierarchy("").top_level_options,
            self._build_signup_hierarchy("").top_level_options,
            self._build_form_hierarchy("").top_level_options,
            self._build_content_hierarchy("").top_level_options,
            self._build_analysis_hierarchy("").top_level_options,
            self._build_tool_use_hierarchy("").top_level_options,
            self._build_unknown_surface_hierarchy("").top_level_options,
        ]
        for opts in common_options:
            for opt in opts:
                self._option_library[opt.name] = opt
