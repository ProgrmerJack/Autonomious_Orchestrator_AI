"""Tests for the RiskGuardian, PolicyMatrix, and RiskCalibration (Phase 4)."""

from __future__ import annotations

import pytest

from agentos_orchestrator.cognition.risk_guardian import (
    RiskAssessment,
    RiskCalibration,
    RiskGuardian,
    assess_action_risk,
    classify_reversibility,
    classify_target,
)
from agentos_orchestrator.os_control.base import UiAction


# ─────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #


@pytest.fixture
def guardian() -> RiskGuardian:
    return RiskGuardian()


def _action(
    action_type: str, selector: str = "element", value: str | None = None
) -> UiAction:
    return UiAction(action_type=action_type, selector=selector, value=value)


# ─────────────────────────────────────────────────────────────────────────── #
# classify_target                                                               #
# ─────────────────────────────────────────────────────────────────────────── #


class TestClassifyTarget:
    def test_payment_button_classified_as_payment(self):
        # "payment" standalone word in selector → matches _PAYMENT_TERMS
        action = _action("click", "confirm-payment-button")
        target_class, sensitivity = classify_target(action)
        assert "payment" in target_class.lower()
        assert sensitivity == "confidential"

    def test_password_field_classified_as_credential(self):
        # "password" in selector → _CREDENTIAL_TERMS match
        action = _action("type", "password-input", value="hunter2")
        target_class, sensitivity = classify_target(action)
        assert "credential" in target_class.lower()
        assert sensitivity == "secret"

    def test_delete_word_in_selector_classified_correctly(self):
        # standalone "delete" word in selector → _FILE_DELETE_TERMS match
        action = _action("click", "delete-btn")
        target_class, _ = classify_target(action)
        assert "delete" in target_class.lower() or "file" in target_class.lower()

    def test_safe_click_returns_ui_public(self):
        # plain selector with no sensitive terms → ui / public
        action = _action("click", "close-btn")
        target_class, sensitivity = classify_target(action)
        assert target_class == "ui"
        assert sensitivity == "public"


# ─────────────────────────────────────────────────────────────────────────── #
# classify_reversibility                                                        #
# ─────────────────────────────────────────────────────────────────────────── #


class TestClassifyReversibility:
    def test_scroll_is_reversible(self):
        action = _action("scroll", "page")
        rev = classify_reversibility(action)
        assert rev == "reversible"

    def test_click_is_hard_to_reverse(self):
        # "click" is in the hard_to_reverse set
        action = _action("click", "menu-item")
        rev = classify_reversibility(action)
        assert rev == "hard_to_reverse"

    def test_delete_action_type_is_irreversible(self):
        # exact action_type "delete" is in the irreversible set
        action = _action("delete", "document.pdf")
        rev = classify_reversibility(action)
        assert rev == "irreversible"

    def test_type_into_form_is_hard_to_reverse(self):
        action = _action("type", "search-box", value="query")
        rev = classify_reversibility(action)
        assert rev in ("reversible", "hard_to_reverse")

    def test_send_action_type_is_irreversible(self):
        # "send" exact match in irreversible set
        action = _action("send", "compose-form")
        rev = classify_reversibility(action)
        assert rev == "irreversible"

    def test_metadata_explicit_override(self):
        action = UiAction(
            action_type="type",
            selector="field",
            value="text",
            metadata={"reversibility": "reversible"},
        )
        rev = classify_reversibility(action)
        assert rev == "reversible"


# ─────────────────────────────────────────────────────────────────────────── #
# RiskGuardian.assess (all params after action are keyword-only)               #
# ─────────────────────────────────────────────────────────────────────────── #


class TestRiskGuardianAssess:
    def test_scroll_has_low_risk(self, guardian):
        action = _action("scroll", "page")
        assessment = guardian.assess(action, objective="Browse the page")
        assert assessment.adjusted_risk < 0.15

    def test_submit_payment_has_high_risk(self, guardian):
        # "submit" → irreversible; "payment-checkout-form" → payment target
        # PolicyMatrix: ("*","payment","irreversible","confidential", 0.88) → matches
        action = _action("submit", "payment-checkout-form")
        assessment = guardian.assess(action, objective="Pay the vendor")
        assert assessment.adjusted_risk >= 0.70

    def test_submit_payment_requires_approval(self, guardian):
        action = _action("submit", "payment-checkout-form")
        assessment = guardian.assess(action, objective="Pay")
        assert assessment.approval_required is True

    def test_delete_action_type_requires_approval(self, guardian):
        # "delete" → irreversible; "important-folder" → file_delete target class
        action = _action("delete", "important-folder")
        assessment = guardian.assess(action, objective="Delete old files")
        assert assessment.approval_required is True

    def test_snapshot_has_minimal_risk(self, guardian):
        # "snapshot" → reversible → wildcard row: base_risk=0.10
        action = _action("snapshot", "desktop")
        assessment = guardian.assess(action, objective="Take a screenshot")
        assert assessment.adjusted_risk < 0.20

    def test_assessment_has_risk_level_string(self, guardian):
        action = _action("click", "btn")
        assessment = guardian.assess(action, objective="Click a button")
        assert assessment.risk_level in (
            "negligible",
            "low",
            "medium",
            "high",
            "critical",
        )

    def test_context_adjustments_is_dict(self, guardian):
        action = _action("submit", "payment-checkout-form")
        assessment = guardian.assess(action, objective="Pay the invoice")
        assert isinstance(assessment.context_adjustments, dict)

    def test_assess_returns_risk_assessment_type(self, guardian):
        action = _action("click", "btn")
        assessment = guardian.assess(action)
        assert isinstance(assessment, RiskAssessment)

    def test_objective_recorded_in_evidence(self, guardian):
        action = _action("click", "btn")
        assessment = guardian.assess(action, objective="Test goal")
        assert "objective" in assessment.evidence


# ─────────────────────────────────────────────────────────────────────────── #
# RiskGuardian.is_safe (keyword-only after action)                             #
# ─────────────────────────────────────────────────────────────────────────── #


class TestRiskGuardianIsSafe:
    def test_scroll_is_safe(self, guardian):
        action = _action("scroll", "main-panel")
        assert guardian.is_safe(action, objective="Scroll down")

    def test_delete_action_is_not_safe_by_default(self, guardian):
        # "delete" → irreversible → file_delete target → base_risk=0.94 ≥ ABORT_THRESHOLD(0.92)
        action = _action("delete", "system-files")
        assert not guardian.is_safe(action, objective="Delete files")

    def test_custom_abort_threshold_permissive(self, guardian):
        # submit to payment with 0.99 threshold → 0.88 < 0.99 → safe
        action = _action("submit", "payment-checkout-form")
        assert guardian.is_safe(action, objective="Pay vendor", abort_threshold=0.99)

    def test_type_text_is_safe_default(self, guardian):
        action = _action("type", "search-box", value="hello")
        # "type" → hard_to_reverse → generic row 0.28-0.35 < 0.92 → safe
        assert guardian.is_safe(action, objective="Search")


# ─────────────────────────────────────────────────────────────────────────── #
# RiskCalibration                                                               #
# ─────────────────────────────────────────────────────────────────────────── #


class TestRiskCalibration:
    def test_record_outcome_increases_delta_on_false_negative(self):
        cal = RiskCalibration()
        initial = cal.delta_for("payment")
        cal.record_outcome("payment", predicted_risk=0.3, actual_incident=True)
        assert cal.delta_for("payment") > initial

    def test_record_outcome_decreases_delta_on_false_positive(self):
        cal = RiskCalibration()
        cal.record_outcome("payment", predicted_risk=0.3, actual_incident=True)
        mid = cal.delta_for("payment")
        cal.record_outcome("payment", predicted_risk=0.8, actual_incident=False)
        assert cal.delta_for("payment") < mid

    def test_delta_bounded_above(self):
        cal = RiskCalibration()
        for _ in range(200):
            cal.record_outcome("file_delete", predicted_risk=0.1, actual_incident=True)
        assert cal.delta_for("file_delete") <= 0.15

    def test_delta_bounded_below(self):
        cal = RiskCalibration()
        for _ in range(200):
            cal.record_outcome("ui", predicted_risk=0.9, actual_incident=False)
        assert cal.delta_for("ui") >= -0.15

    def test_unknown_class_returns_zero_delta(self):
        cal = RiskCalibration()
        assert cal.delta_for("totally_unknown_class_xyz") == 0.0

    def test_separate_classes_independent(self):
        cal = RiskCalibration()
        cal.record_outcome("payment", predicted_risk=0.1, actual_incident=True)
        assert cal.delta_for("credential") == 0.0


# ─────────────────────────────────────────────────────────────────────────── #
# Module-level assess_action_risk (keyword-only args)                          #
# ─────────────────────────────────────────────────────────────────────────── #


def test_assess_action_risk_module_level():
    action = _action("click", "home-btn")
    assessment = assess_action_risk(action, objective="Navigate home")
    assert isinstance(assessment, RiskAssessment)
    assert 0.0 <= assessment.adjusted_risk <= 1.0
