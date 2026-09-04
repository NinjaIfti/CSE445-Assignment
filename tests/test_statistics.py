"""Verification of the paired-CV significance machinery used for the CO2 analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_runner import (  # noqa: E402
    OUTER_FOLDS,
    _paired_estimators,
    corrected_resampled_ttest,
    paired_cv_scores,
    render_significance_markdown,
)


# --------------------------------------------------------------------------------------
# Corrected resampled t-test
# --------------------------------------------------------------------------------------
def test_identical_scores_are_not_significant():
    r = corrected_resampled_ttest([0.9] * 10, [0.9] * 10, 10)
    assert r["mean_difference"] == 0.0
    assert r["significant"] is False
    assert r["p_value"] == 1.0


def test_constant_nonzero_difference_is_significant_not_ignored():
    """Zero variance with a non-zero mean is a perfectly consistent win, not a null result."""
    r = corrected_resampled_ttest([0.91] * 10, [0.90] * 10, 10)
    assert r["mean_difference"] == pytest.approx(0.01)
    assert r["significant"] is True
    assert r["p_value"] == 0.0


def test_correction_is_more_conservative_than_the_naive_paired_ttest():
    """The whole point of the correction: CV folds share training data, so the naive
    paired t-test understates the variance and over-claims significance."""
    a = [0.95, 0.92, 0.97, 0.90, 0.94, 0.96, 0.91, 0.93, 0.95, 0.92]
    b = [0.93, 0.91, 0.94, 0.89, 0.93, 0.93, 0.90, 0.92, 0.92, 0.91]

    corrected = corrected_resampled_ttest(a, b, 10)
    naive_p = float(scipy_stats.ttest_rel(a, b).pvalue)

    assert corrected["p_value"] > naive_p
    assert corrected["t_statistic"] < float(abs(scipy_stats.ttest_rel(a, b).statistic))


def test_large_separation_is_detected():
    a = [0.95, 0.92, 0.97, 0.90, 0.94, 0.96, 0.91, 0.93, 0.95, 0.92]
    c = [0.70, 0.72, 0.68, 0.71, 0.69, 0.70, 0.73, 0.67, 0.71, 0.70]
    r = corrected_resampled_ttest(a, c, 10)
    assert r["significant"] is True
    assert r["mean_difference"] > 0.2


def test_the_test_is_antisymmetric():
    a = [0.95, 0.92, 0.97, 0.90, 0.94]
    b = [0.93, 0.91, 0.94, 0.89, 0.93]
    forward = corrected_resampled_ttest(a, b, 5)
    reverse = corrected_resampled_ttest(b, a, 5)
    assert forward["mean_difference"] == pytest.approx(-reverse["mean_difference"])
    assert forward["p_value"] == pytest.approx(reverse["p_value"])


def test_correction_factor_matches_the_published_formula():
    """t = mean(d) / sqrt((1/k + 1/(k-1)) * var(d)), Nadeau & Bengio."""
    import numpy as np

    a = [0.90, 0.95, 0.85, 0.92, 0.88]
    b = [0.88, 0.90, 0.84, 0.89, 0.86]
    k = 5
    d = np.array(a) - np.array(b)
    expected_t = d.mean() / ((1 / k + 1 / (k - 1)) * d.var(ddof=1)) ** 0.5

    assert corrected_resampled_ttest(a, b, k)["t_statistic"] == pytest.approx(
        float(expected_t), rel=1e-3
    )


# --------------------------------------------------------------------------------------
# Paired cross-validation
# --------------------------------------------------------------------------------------
def test_paired_estimators_are_the_three_compared_algorithms():
    est = _paired_estimators()
    assert list(est) == ["Random Forest", "Kernel SVM (tuned)", "Deep MLP (Dropout+BN)"]


def test_paired_cv_scores_uses_identical_folds_for_every_algorithm():
    scores = paired_cv_scores("iris", folds=3)
    assert list(scores) == ["Random Forest", "Kernel SVM (tuned)", "Deep MLP (Dropout+BN)"]
    for label, fold_scores in scores.items():
        assert len(fold_scores) == 3, label
        assert all(0.0 <= s <= 1.0 for s in fold_scores), label


def test_significance_markdown_renders_a_table_per_dataset():
    scores = {"iris": {
        "Random Forest": [0.9, 0.95, 0.92],
        "Kernel SVM (tuned)": [0.91, 0.96, 0.93],
        "Deep MLP (Dropout+BN)": [0.88, 0.90, 0.89],
    }}
    md = render_significance_markdown(scores)
    assert "### iris" in md
    assert "Random Forest vs Kernel SVM (tuned)" in md
    assert "Nadeau-Bengio" in md
    # Three algorithms -> three pairwise comparisons.
    assert md.count(" vs ") == 3
