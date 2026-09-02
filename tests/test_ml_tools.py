"""Offline verification of the ML tool layer. Requires no Ollama and no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ml_tools  # noqa: E402
from ml_tools import DATASETS, TOOL_SPECS, call_tool, describe_tools  # noqa: E402


def parse(result: str) -> dict:
    """Every tool must return a JSON string -- that is the contract with the LLM."""
    assert isinstance(result, str)
    return json.loads(result)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
def test_registry_exposes_six_tools():
    assert len(TOOL_SPECS) == 6
    assert set(TOOL_SPECS) == {
        "load_dataset_summary", "train_sklearn_model", "train_pytorch_mlp",
        "tune_hyperparameters", "feature_selection", "train_deep_classifier",
    }


def test_describe_tools_lists_every_tool_with_an_example():
    catalogue = describe_tools()
    for name, spec in TOOL_SPECS.items():
        assert name in catalogue
        assert json.dumps(spec["example"]) in catalogue


# --------------------------------------------------------------------------------------
# Task 1 -- baseline tools
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_load_dataset_summary_all_datasets(dataset):
    out = parse(call_tool("load_dataset_summary", {"dataset_name": dataset}))
    assert out["status"] == "ok"
    assert out["dataset"] == dataset
    assert out["n_samples"] > 0 and out["n_features"] > 0
    assert out["missing_values"] == 0
    assert sum(out["class_balance"].values()) == out["n_samples"]


@pytest.mark.parametrize("model", ["decision_tree", "logistic_regression", "random_forest"])
def test_train_sklearn_model_reports_cv_statistics(model):
    out = parse(call_tool("train_sklearn_model",
                          {"dataset_name": "iris", "model_type": model}))
    assert out["status"] == "ok"
    assert 0.0 <= out["test_accuracy"] <= 1.0
    assert 0.0 <= out["cv_mean_accuracy"] <= 1.0
    assert out["cv_std"] >= 0.0
    assert len(out["cv_95_interval"]) == 2


def test_train_pytorch_mlp_learns_iris():
    out = parse(call_tool("train_pytorch_mlp",
                          {"dataset_name": "iris", "epochs": 120, "hidden_dim": 32}))
    assert out["status"] == "ok"
    assert out["framework"] == "PyTorch"
    assert out["test_accuracy"] > 0.8, "an MLP should comfortably beat chance on iris"
    assert out["n_parameters"] > 0


def test_results_are_deterministic_across_calls():
    first = parse(call_tool("train_pytorch_mlp", {"dataset_name": "wine", "epochs": 40}))
    second = parse(call_tool("train_pytorch_mlp", {"dataset_name": "wine", "epochs": 40}))
    assert first["test_accuracy"] == second["test_accuracy"]
    assert first["final_loss"] == second["final_loss"]


# --------------------------------------------------------------------------------------
# Task 2 -- advanced tools
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("search_type", ["grid", "random"])
def test_tune_hyperparameters_decision_tree(search_type):
    out = parse(call_tool("tune_hyperparameters", {
        "dataset_name": "iris", "model_type": "decision_tree",
        "search_type": search_type, "n_iter": 6,
    }))
    assert out["status"] == "ok"
    assert out["search_type"] == search_type
    assert set(out["best_params"]) <= {
        "max_depth", "min_samples_split", "min_samples_leaf", "criterion"}
    assert 0.0 <= out["best_cv_accuracy"] <= 1.0
    assert len(out["leaderboard_top3"]) == 3


def test_tune_hyperparameters_accepts_svm_alias_for_kernel_svc():
    out = parse(call_tool("tune_hyperparameters", {
        "dataset_name": "iris", "model_type": "svm",
        "search_type": "random", "n_iter": 5,
    }))
    assert out["status"] == "ok"
    assert out["model"] == "svc"
    assert "kernel" in out["best_params"]


def test_feature_selection_pca_and_sfs():
    out = parse(call_tool("feature_selection", {
        "dataset_name": "wine", "method": "both", "n_components": 3,
    }))
    assert out["status"] == "ok"
    assert out["n_features_original"] == 13
    assert len(out["pca"]["explained_variance_ratio"]) == 3
    assert 0.0 < out["pca"]["cumulative_explained_variance"] <= 1.0
    assert out["sequential_feature_selection"]["n_features_selected"] == 3
    assert len(out["sequential_feature_selection"]["selected_features"]) == 3


def test_deep_classifier_uses_dropout_batchnorm_and_scheduler():
    out = parse(call_tool("train_deep_classifier", {
        "dataset_name": "breast_cancer", "hidden_dims": [64, 32],
        "dropout": 0.3, "batch_norm": True, "epochs": 40, "scheduler": "cosine",
    }))
    assert out["status"] == "ok"
    assert out["architecture"]["dropout"] == 0.3
    assert out["architecture"]["batch_norm"] is True
    assert out["optimisation"]["scheduler"] == "cosine"
    assert out["test_accuracy"] > 0.85
    # Cosine annealing must actually decay the learning rate over training.
    assert out["lr_history"][-1]["lr"] < out["lr_history"][0]["lr"]


@pytest.mark.parametrize("scheduler", ["cosine", "step", "plateau", "none"])
def test_every_scheduler_runs(scheduler):
    out = parse(call_tool("train_deep_classifier", {
        "dataset_name": "iris", "hidden_dims": [16], "epochs": 12, "scheduler": scheduler,
    }))
    assert out["status"] == "ok"
    assert out["optimisation"]["scheduler"] == scheduler


def test_deep_classifier_without_batchnorm_still_trains():
    out = parse(call_tool("train_deep_classifier", {
        "dataset_name": "iris", "hidden_dims": [16], "epochs": 15, "batch_norm": False,
    }))
    assert out["status"] == "ok"
    assert out["architecture"]["batch_norm"] is False


# --------------------------------------------------------------------------------------
# Task 3 -- structured, recoverable errors
# --------------------------------------------------------------------------------------
def test_unknown_dataset_is_recoverable_not_fatal():
    out = parse(call_tool("load_dataset_summary", {"dataset_name": "titanic"}))
    assert out["status"] == "error"
    assert out["error_type"] == "unknown_dataset"
    assert "retry_with" in out


def test_dataset_alias_is_normalised_and_recorded():
    out = parse(call_tool("load_dataset_summary", {"dataset_name": "BreastCancer"}))
    assert out["status"] == "ok"
    assert out["dataset"] == "breast_cancer"


def test_argument_alias_is_normalised_and_reported():
    out = parse(call_tool("train_sklearn_model",
                          {"dataset": "iris", "model": "rf"}))
    assert out["status"] == "ok"
    assert out["_normalised_arguments"] == {"dataset": "dataset_name", "model": "model_type"}


def test_unknown_tool_lists_the_valid_tools():
    out = parse(call_tool("train_xgboost", {"dataset_name": "iris"}))
    assert out["error_type"] == "unknown_tool"
    assert "load_dataset_summary" in out["hint"]


def test_unexpected_argument_is_rejected_with_the_valid_signature():
    out = parse(call_tool("load_dataset_summary",
                          {"dataset_name": "iris", "n_estimators": 100}))
    assert out["error_type"] == "unexpected_argument"
    assert "n_estimators" in out["message"]


def test_missing_required_argument_offers_an_example():
    out = parse(call_tool("train_sklearn_model", {"dataset_name": "iris"}))
    assert out["error_type"] == "missing_argument"
    assert out["retry_with"]["model_type"] == "random_forest"


def test_pca_shape_mismatch_suggests_a_valid_dimensionality():
    out = parse(call_tool("feature_selection",
                          {"dataset_name": "iris", "method": "pca", "n_components": 99}))
    assert out["error_type"] == "shape_mismatch"
    assert out["retry_with"]["n_components"] == 4  # iris has exactly 4 features


def test_out_of_range_learning_rate_is_rejected():
    out = parse(call_tool("train_pytorch_mlp", {"dataset_name": "iris", "lr": 50}))
    assert out["error_type"] == "invalid_parameter"


def test_unsupported_model_for_tool_points_at_the_right_tool():
    out = parse(call_tool("train_sklearn_model",
                          {"dataset_name": "iris", "model_type": "svc"}))
    assert out["error_type"] == "unknown_model"
    assert "tune_hyperparameters" in out["hint"]


def test_nan_loss_is_reported_as_a_recoverable_error(monkeypatch):
    """Force divergence so the nan_loss self-healing branch is genuinely exercised."""
    import torch

    real_cross_entropy = ml_tools.nn.CrossEntropyLoss

    class ExplodingLoss(real_cross_entropy):
        def forward(self, inputs, target):
            return torch.tensor(float("nan"))

    monkeypatch.setattr(ml_tools.nn, "CrossEntropyLoss", ExplodingLoss)
    out = parse(call_tool("train_pytorch_mlp",
                          {"dataset_name": "iris", "epochs": 5, "lr": 0.5}))
    assert out["error_type"] == "nan_loss"
    assert out["retry_with"]["lr"] == pytest.approx(0.05)
    assert "smaller" in out["hint"]


def test_every_tool_returns_valid_json_for_a_garbage_dataset():
    for name in TOOL_SPECS:
        out = parse(call_tool(name, {"dataset_name": "not_a_dataset"}))
        assert out["status"] == "error"
        assert "error_type" in out
