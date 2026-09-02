"""
ml_tools.py -- Machine Learning tool registry for the CSE445 Autonomous Local LLM Agent.

Every public tool is a pure function that takes JSON-friendly scalars and returns a
JSON *string*. That contract is what makes them safely callable by a small local LLM:
the model never touches Python objects, only text in and text out.

Tools are grouped as follows.

  Baseline (Assignment Task 1)
    1. load_dataset_summary   -- exploratory data analysis
    2. train_sklearn_model    -- decision tree / logistic regression / random forest
    3. train_pytorch_mlp      -- plain 1-hidden-layer MLP

  Advanced (Assignment Task 2)
    4. tune_hyperparameters   -- GridSearchCV / RandomizedSearchCV over SVC + trees
    5. feature_selection      -- PCA and Sequential Feature Selection
    6. train_deep_classifier  -- deep MLP with Dropout, BatchNorm and LR schedulers

Error handling (Assignment Task 3)
    Tools never raise for *recoverable* problems. They return a structured error
    envelope containing a machine-readable `error_type`, a human `message`, a `hint`
    and -- where possible -- a concrete `retry_with` dictionary. The ReAct controller
    in react_agent.py consumes those fields to self-heal and retry.
"""

from __future__ import annotations

import inspect
import json
import time
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def _seed_everything(seed: int = RANDOM_STATE) -> None:
    """Re-seed before every stochastic tool so repeated agent calls are deterministic."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------------------
# Dataset registry
# --------------------------------------------------------------------------------------
DATASETS: Dict[str, Callable[[], Any]] = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

# Small local LLMs routinely invent dataset aliases. Map the common ones instead of
# failing, and record the normalisation in the response so the trace stays honest.
DATASET_ALIASES: Dict[str, str] = {
    "breastcancer": "breast_cancer",
    "breast-cancer": "breast_cancer",
    "cancer": "breast_cancer",
    "wisconsin": "breast_cancer",
    "load_iris": "iris",
    "iris_dataset": "iris",
    "wines": "wine",
}


# --------------------------------------------------------------------------------------
# Response envelopes
# --------------------------------------------------------------------------------------
def _ok(payload: Dict[str, Any]) -> str:
    return json.dumps({"status": "ok", **payload})


def _err(error_type: str, message: str, hint: str = "", retry_with: Dict | None = None) -> str:
    """Structured, recoverable failure. `error_type` drives self-correction in the agent."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    if hint:
        envelope["hint"] = hint
    if retry_with:
        envelope["retry_with"] = retry_with
    return json.dumps(envelope)


def _resolve_dataset(dataset_name: Any):
    """Return (bunch, canonical_name, error_json). Exactly one of bunch/error is None."""
    if not isinstance(dataset_name, str):
        return None, None, _err(
            "invalid_parameter",
            f"dataset_name must be a string, got {type(dataset_name).__name__}.",
            f"Pass one of {list(DATASETS)} as a quoted string.",
        )

    name = dataset_name.lower().strip().replace(" ", "_")
    name = DATASET_ALIASES.get(name, name)

    if name not in DATASETS:
        return None, None, _err(
            "unknown_dataset",
            f"Unknown dataset '{dataset_name}'.",
            f"Valid dataset_name values are exactly: {list(DATASETS)}.",
            {"dataset_name": "iris"},
        )
    return DATASETS[name](), name, None


def _split(data, test_size: float = 0.2):
    return train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )


def _check_unit_interval(value: Any, field: str, lo: float, hi: float):
    """Validate a float hyperparameter, returning an error JSON string or None."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return _err(
            "invalid_parameter",
            f"{field} must be a number, got {type(value).__name__}.",
            f"Pass {field} as a number strictly between {lo} and {hi}.",
        )
    if not (lo < float(value) < hi):
        return _err(
            "invalid_parameter",
            f"{field}={value} is out of range.",
            f"{field} must be strictly between {lo} and {hi}.",
            {field: 0.2 if field == "test_size" else value},
        )
    return None


# ======================================================================================
# TASK 1 -- Baseline tools
# ======================================================================================
def load_dataset_summary(dataset_name: str) -> str:
    """Load a benchmark dataset and return summary statistics for exploratory analysis."""
    data, name, err = _resolve_dataset(dataset_name)
    if err:
        return err

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    classes, counts = np.unique(data.target, return_counts=True)
    return _ok(
        {
            "dataset": name,
            "n_samples": int(df.shape[0]),
            "n_features": int(len(data.feature_names)),
            "feature_names": list(data.feature_names)[:10],
            "n_classes": int(len(classes)),
            "class_names": [str(c) for c in getattr(data, "target_names", classes)],
            "class_balance": {str(c): int(n) for c, n in zip(classes, counts)},
            "missing_values": int(df.isnull().sum().sum()),
            "feature_scale_range": {
                "min_of_means": round(float(df[data.feature_names].mean().min()), 4),
                "max_of_means": round(float(df[data.feature_names].mean().max()), 4),
            },
        }
    )


SKLEARN_MODELS = ("decision_tree", "logistic_regression", "random_forest")

MODEL_ALIASES = {
    "tree": "decision_tree",
    "decisiontree": "decision_tree",
    "dt": "decision_tree",
    "logreg": "logistic_regression",
    "logistic": "logistic_regression",
    "logisticregression": "logistic_regression",
    "rf": "random_forest",
    "randomforest": "random_forest",
    "forest": "random_forest",
    "svm": "svc",
    "kernel_svm": "svc",
    "support_vector_machine": "svc",
}


def _normalise_model(model_type: Any) -> str:
    if not isinstance(model_type, str):
        return ""
    key = model_type.lower().strip().replace(" ", "_").replace("-", "_")
    return MODEL_ALIASES.get(key, key)


def train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> str:
    """Train a Scikit-Learn classifier and report hold-out plus 5-fold CV accuracy."""
    data, name, err = _resolve_dataset(dataset_name)
    if err:
        return err

    range_err = _check_unit_interval(test_size, "test_size", 0.0, 1.0)
    if range_err:
        return range_err

    model = _normalise_model(model_type)
    if model not in SKLEARN_MODELS:
        return _err(
            "unknown_model",
            f"Unsupported model_type '{model_type}' for train_sklearn_model.",
            f"model_type must be one of {list(SKLEARN_MODELS)}. "
            "For SVC / kernel SVM use the tune_hyperparameters tool instead.",
            {"dataset_name": name, "model_type": "random_forest"},
        )

    _seed_everything()
    X_train, X_test, y_train, y_test = _split(data, float(test_size))

    if model == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    elif model == "logistic_regression":
        # Scaling matters on breast_cancer, whose features span four orders of magnitude.
        clf = Pipeline(
            [("scaler", StandardScaler()),
             ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
        )
    else:
        clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)

    start = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, data.data, data.target, cv=cv)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return _ok(
        {
            "framework": "scikit-learn",
            "model": model,
            "dataset": name,
            "test_size": float(test_size),
            "test_accuracy": round(float(acc), 4),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "cv_95_interval": [
                round(float(cv_scores.mean() - 1.96 * cv_scores.std()), 4),
                round(float(cv_scores.mean() + 1.96 * cv_scores.std()), 4),
            ],
            "fit_seconds": round(fit_seconds, 4),
        }
    )


def train_pytorch_mlp(
    dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01,
    seed: int = RANDOM_STATE,
) -> str:
    """Train a single-hidden-layer PyTorch MLP (full-batch) on a classification dataset.

    `seed` controls weight initialisation only; the train/test split is held fixed, so
    repeating a call across seeds isolates optimisation variance from split variance.
    """
    data, name, err = _resolve_dataset(dataset_name)
    if err:
        return err

    for field, value, lo, hi in (("hidden_dim", hidden_dim, 1, 4097), ("epochs", epochs, 1, 5001)):
        if not isinstance(value, int) or isinstance(value, bool) or not (lo <= value < hi):
            return _err(
                "invalid_parameter",
                f"{field}={value!r} is invalid.",
                f"{field} must be an integer in [{lo}, {hi - 1}].",
                {"dataset_name": name, field: 32 if field == "hidden_dim" else 50},
            )
    lr_err = _check_unit_interval(lr, "lr", 0.0, 10.0)
    if lr_err:
        return lr_err

    _seed_everything(int(seed))
    X_train, X_test, y_train, y_test = _split(data)

    # Standardise on TRAIN statistics only -- applying test statistics would leak.
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_features = X_train.shape[1]
    num_classes = int(len(np.unique(data.target)))

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(lr))

    start = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        if not torch.isfinite(loss):
            # Task 3 self-healing hook: surface divergence as a recoverable error.
            return _err(
                "nan_loss",
                f"Loss diverged to {loss.item()} at epoch {epoch} with lr={lr}.",
                "The learning rate is too high. Retry with a 10x smaller lr.",
                {"dataset_name": name, "hidden_dim": hidden_dim, "epochs": epochs,
                 "lr": round(float(lr) / 10, 6)},
            )
        loss.backward()
        optimizer.step()
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(model(X_val_t), dim=1)
        acc = (test_preds == y_val_t).float().mean().item()

    return _ok(
        {
            "framework": "PyTorch",
            "model": "mlp_1_hidden_layer",
            "dataset": name,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "lr": float(lr),
            "seed": int(seed),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "final_loss": round(float(loss.item()), 4),
            "test_accuracy": round(float(acc), 4),
            "train_seconds": round(train_seconds, 4),
        }
    )


# --------------------------------------------------------------------------------------
# Small helpers shared by the advanced tools
# --------------------------------------------------------------------------------------
def _strip_prefix(key: str) -> str:
    """Drop the sklearn Pipeline step prefix so the LLM sees plain hyperparameter names."""
    return key.split("__", 1)[-1]


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars to built-in types so json.dumps never chokes."""
    if isinstance(value, np.generic):
        return value.item()
    return value


# ======================================================================================
# TASK 2 -- Advanced ML tool expansion
# ======================================================================================
TUNABLE_MODELS = ("svc", "decision_tree")

# Kept module-level so the report and the tests can quote the exact search space.
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "svc": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "clf__kernel": ["rbf", "poly", "linear"],
    },
    "decision_tree": {
        "clf__max_depth": [2, 3, 4, 6, 8, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__criterion": ["gini", "entropy"],
    },
}


def tune_hyperparameters(
    dataset_name: str,
    model_type: str,
    search_type: str = "grid",
    cv: int = 5,
    n_iter: int = 20,
    scoring: str = "accuracy",
) -> str:
    """Run GridSearchCV or RandomizedSearchCV over an SVC (kernel SVM) or Decision Tree."""
    data, name, err = _resolve_dataset(dataset_name)
    if err:
        return err

    model = _normalise_model(model_type)
    if model not in TUNABLE_MODELS:
        return _err(
            "unknown_model",
            f"tune_hyperparameters does not support model_type {model_type!r}.",
            f"model_type must be one of {list(TUNABLE_MODELS)}; "
            "svm and kernel_svm are accepted aliases for svc.",
            {"dataset_name": name, "model_type": "svc", "search_type": search_type},
        )

    search_kind = str(search_type).lower().strip()
    if search_kind in ("randomized", "randomised", "random_search"):
        search_kind = "random"
    if search_kind not in ("grid", "random"):
        return _err(
            "invalid_parameter",
            f"search_type {search_type!r} is not recognised.",
            "search_type must be either grid or random.",
            {"dataset_name": name, "model_type": model, "search_type": "grid"},
        )

    if not isinstance(cv, int) or isinstance(cv, bool) or not (2 <= cv <= 10):
        return _err(
            "invalid_parameter",
            f"cv={cv!r} is invalid.",
            "cv must be an integer between 2 and 10.",
            {"dataset_name": name, "model_type": model, "cv": 5},
        )

    _seed_everything()
    X_train, X_test, y_train, y_test = _split(data)

    # SVC is scale-sensitive; the scaler lives inside the pipeline so it is refit on
    # every CV fold rather than leaking test-fold statistics into training.
    estimator = (
        SVC(random_state=RANDOM_STATE)
        if model == "svc"
        else DecisionTreeClassifier(random_state=RANDOM_STATE)
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    grid = PARAM_GRIDS[model]

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    if search_kind == "grid":
        search = GridSearchCV(pipe, grid, cv=splitter, scoring=scoring, n_jobs=-1)
        n_candidates = int(np.prod([len(v) for v in grid.values()]))
    else:
        n_candidates = min(int(n_iter), int(np.prod([len(v) for v in grid.values()])))
        search = RandomizedSearchCV(
            pipe,
            grid,
            n_iter=n_candidates,
            cv=splitter,
            scoring=scoring,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    start = time.perf_counter()
    try:
        search.fit(X_train, y_train)
    except Exception as exc:  # e.g. an unsupported scoring string invented by the LLM
        return _err(
            "search_failed",
            f"{type(exc).__name__}: {exc}",
            "Retry with scoring set to accuracy and the default search space.",
            {
                "dataset_name": name,
                "model_type": model,
                "search_type": search_kind,
                "scoring": "accuracy",
            },
        )
    search_seconds = time.perf_counter() - start

    test_acc = accuracy_score(y_test, search.best_estimator_.predict(X_test))

    # Top 3 configurations, so the agent can reason about how sensitive the search was.
    results = pd.DataFrame(search.cv_results_)
    top = results.nsmallest(3, "rank_test_score")[
        ["params", "mean_test_score", "std_test_score"]
    ]
    leaderboard = [
        {
            "params": {_strip_prefix(k): _jsonable(v) for k, v in row.params.items()},
            "cv_mean": round(float(row.mean_test_score), 4),
            "cv_std": round(float(row.std_test_score), 4),
        }
        for row in top.itertuples()
    ]

    return _ok(
        {
            "tool": "tune_hyperparameters",
            "dataset": name,
            "model": model,
            "search_type": search_kind,
            "cv_folds": cv,
            "n_candidates_evaluated": n_candidates,
            "best_params": {
                _strip_prefix(k): _jsonable(v) for k, v in search.best_params_.items()
            },
            "best_cv_accuracy": round(float(search.best_score_), 4),
            "test_accuracy": round(float(test_acc), 4),
            "leaderboard_top3": leaderboard,
            "search_seconds": round(search_seconds, 3),
        }
    )


def feature_selection(
    dataset_name: str,
    method: str = "pca",
    n_components: int = 2,
    direction: str = "forward",
) -> str:
    """Reduce dimensionality with PCA and/or Sequential Feature Selection, then score it.

    Both branches are compared against an identical full-feature baseline so the agent
    can quantify the accuracy cost of the compression rather than guessing at it.
    """
    data, name, err = _resolve_dataset(dataset_name)
    if err:
        return err

    method_kind = str(method).lower().strip()
    method_aliases = {
        "sequential": "sfs",
        "sequential_feature_selection": "sfs",
        "forward_selection": "sfs",
        "all": "both",
    }
    method_kind = method_aliases.get(method_kind, method_kind)
    if method_kind not in ("pca", "sfs", "both"):
        return _err(
            "invalid_parameter",
            f"method {method!r} is not recognised.",
            "method must be pca, sfs (sequential feature selection), or both.",
            {"dataset_name": name, "method": "pca", "n_components": n_components},
        )

    n_features = int(data.data.shape[1])
    if not isinstance(n_components, int) or isinstance(n_components, bool):
        return _err(
            "invalid_parameter",
            f"n_components must be an integer, got {type(n_components).__name__}.",
            f"Pass an integer between 1 and {n_features}.",
            {"dataset_name": name, "method": method_kind, "n_components": 2},
        )
    if not (1 <= n_components <= n_features):
        # The classic dimensionality shape error the agent is required to recover from.
        return _err(
            "shape_mismatch",
            f"n_components={n_components} is invalid for {name}, which has "
            f"only {n_features} features.",
            f"n_components must satisfy 1 <= n_components <= {n_features}.",
            {
                "dataset_name": name,
                "method": method_kind,
                "n_components": min(max(1, n_components), n_features),
            },
        )

    _seed_everything()
    X_train, X_test, y_train, y_test = _split(data)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    def _score(xtr, xte) -> float:
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        clf.fit(xtr, y_train)
        return float(accuracy_score(y_test, clf.predict(xte)))

    baseline_acc = _score(X_train_s, X_test_s)
    payload: Dict[str, Any] = {
        "tool": "feature_selection",
        "dataset": name,
        "method": method_kind,
        "n_features_original": n_features,
        "baseline_accuracy_all_features": round(baseline_acc, 4),
    }

    if method_kind in ("pca", "both"):
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X_train_s)
        evr = pca.explained_variance_ratio_
        payload["pca"] = {
            "n_components": n_components,
            "explained_variance_ratio": [round(float(v), 4) for v in evr],
            "cumulative_explained_variance": round(float(evr.sum()), 4),
            "accuracy": round(_score(pca.transform(X_train_s), pca.transform(X_test_s)), 4),
            "compression_ratio": round(n_components / n_features, 4),
        }

    if method_kind in ("sfs", "both"):
        sfs_direction = str(direction).lower().strip()
        if sfs_direction not in ("forward", "backward"):
            return _err(
                "invalid_parameter",
                f"direction {direction!r} is not recognised.",
                "direction must be forward or backward.",
                {
                    "dataset_name": name,
                    "method": method_kind,
                    "n_components": n_components,
                    "direction": "forward",
                },
            )
        sfs = SequentialFeatureSelector(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            n_features_to_select=n_components,
            direction=sfs_direction,
            cv=3,
            n_jobs=-1,
        ).fit(X_train_s, y_train)
        mask = sfs.get_support()
        payload["sequential_feature_selection"] = {
            "direction": sfs_direction,
            "n_features_selected": int(mask.sum()),
            "selected_features": [str(f) for f in np.array(data.feature_names)[mask]],
            "accuracy": round(_score(sfs.transform(X_train_s), sfs.transform(X_test_s)), 4),
        }

    return _ok(payload)


class DeepClassifier(nn.Module):
    """Configurable MLP: [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # noqa: D102
        return self.net(x)


def train_deep_classifier(
    dataset_name: str,
    hidden_dims: Any = (64, 32),
    dropout: float = 0.3,
    batch_norm: bool = True,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 32,
    scheduler: str = "cosine",
    weight_decay: float = 0.0,
    seed: int = RANDOM_STATE,
) -> str:
    """Train a regularised deep MLP with Dropout, BatchNorm and a learning-rate scheduler.

    `seed` controls weight initialisation and batch shuffling only; the train/test split is
    held fixed, so repeating a call across seeds isolates optimisation variance.
    """
    data, name, err = _resolve_dataset(dataset_name)
    if err:
        return err

    # hidden_dims arrives from JSON as a list, but a small LLM may send 64 or "64,32".
    if isinstance(hidden_dims, (int, float)) and not isinstance(hidden_dims, bool):
        hidden_dims = [int(hidden_dims)]
    elif isinstance(hidden_dims, str):
        try:
            hidden_dims = [
                int(p) for p in hidden_dims.replace("[", "").replace("]", "").split(",")
            ]
        except ValueError:
            return _err(
                "invalid_parameter",
                f"hidden_dims={hidden_dims!r} could not be parsed.",
                "Pass hidden_dims as a JSON list of integers, for example [64, 32].",
                {"dataset_name": name, "hidden_dims": [64, 32]},
            )
    hidden_dims = list(hidden_dims or [])
    if not hidden_dims or any(
        not isinstance(h, int) or isinstance(h, bool) or h < 1 for h in hidden_dims
    ):
        return _err(
            "invalid_parameter",
            f"hidden_dims={hidden_dims!r} must be a non-empty list of positive integers.",
            "For example [64, 32] gives a two-hidden-layer network.",
            {"dataset_name": name, "hidden_dims": [64, 32]},
        )

    dropout_err = _check_unit_interval(dropout, "dropout", -0.0001, 1.0)
    if dropout_err:
        return dropout_err
    lr_err = _check_unit_interval(lr, "lr", 0.0, 10.0)
    if lr_err:
        return lr_err

    sched_kind = str(scheduler).lower().strip()
    sched_aliases = {
        "cosine_annealing": "cosine",
        "cosineannealinglr": "cosine",
        "steplr": "step",
        "reduce_on_plateau": "plateau",
        "reducelronplateau": "plateau",
        "null": "none",
    }
    sched_kind = sched_aliases.get(sched_kind, sched_kind)
    if sched_kind not in ("cosine", "step", "plateau", "none"):
        return _err(
            "invalid_parameter",
            f"scheduler {scheduler!r} is not recognised.",
            "scheduler must be one of cosine, step, plateau, none.",
            {"dataset_name": name, "scheduler": "cosine"},
        )

    _seed_everything(int(seed))
    X_train, X_test, y_train, y_test = _split(data)
    scaler = StandardScaler().fit(X_train)
    X_t = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.long)

    n_train = X_t.shape[0]
    batch_size = int(max(2, min(int(batch_size), n_train)))  # BatchNorm needs >1 sample
    num_classes = int(len(np.unique(data.target)))

    model = DeepClassifier(
        X_t.shape[1], hidden_dims, num_classes, float(dropout), bool(batch_norm)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    if sched_kind == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    elif sched_kind == "step":
        sched = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, int(epochs) // 3), gamma=0.1
        )
    elif sched_kind == "plateau":
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    else:
        sched = None

    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2 and batch_norm:
                continue  # a trailing single-sample batch would break BatchNorm1d
            optimizer.zero_grad()
            loss = criterion(model(X_t[idx]), y_t[idx])
            if not torch.isfinite(loss):
                return _err(
                    "nan_loss",
                    f"Loss became {loss.item()} at epoch {epoch} with lr={lr}, "
                    f"weight_decay={weight_decay}.",
                    "Training diverged. Retry with a 10x smaller lr, and consider "
                    "scheduler set to plateau.",
                    {
                        "dataset_name": name,
                        "hidden_dims": hidden_dims,
                        "dropout": dropout,
                        "epochs": epochs,
                        "lr": round(float(lr) / 10, 6),
                    },
                )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        mean_loss = epoch_loss / max(1, n_batches)
        if sched is not None:
            if sched_kind == "plateau":
                sched.step(mean_loss)
            else:
                sched.step()
        if epoch % max(1, int(epochs) // 5) == 0 or epoch == int(epochs) - 1:
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(mean_loss, 4),
                    "lr": round(float(optimizer.param_groups[0]["lr"]), 6),
                }
            )
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        train_acc = (torch.argmax(model(X_t), 1) == y_t).float().mean().item()
        test_acc = (torch.argmax(model(X_val_t), 1) == y_val_t).float().mean().item()

    return _ok(
        {
            "framework": "PyTorch",
            "model": "deep_mlp_regularised",
            "dataset": name,
            "architecture": {
                "hidden_dims": hidden_dims,
                "dropout": float(dropout),
                "batch_norm": bool(batch_norm),
                "n_parameters": int(sum(p.numel() for p in model.parameters())),
            },
            "optimisation": {
                "epochs": int(epochs),
                "lr": float(lr),
                "batch_size": batch_size,
                "scheduler": sched_kind,
                "weight_decay": float(weight_decay),
                "seed": int(seed),
            },
            "final_loss": history[-1]["loss"] if history else None,
            "final_lr": history[-1]["lr"] if history else float(lr),
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "generalisation_gap": round(float(train_acc - test_acc), 4),
            "lr_history": history,
            "train_seconds": round(train_seconds, 4),
        }
    )


# ======================================================================================
# Tool registry and dispatcher
# ======================================================================================
# The registry is the single source of truth: the ReAct system prompt in react_agent.py
# is generated from it, so the tool list the LLM sees can never drift from the code.
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "load_dataset_summary": {
        "func": load_dataset_summary,
        "description": "Return summary statistics (shape, classes, balance, missing values) "
                       "for a benchmark dataset.",
        "example": {"dataset_name": "iris"},
    },
    "train_sklearn_model": {
        "func": train_sklearn_model,
        "description": "Train a Scikit-Learn classifier (decision_tree, logistic_regression, "
                       "random_forest) and return hold-out accuracy plus 5-fold CV mean/std.",
        "example": {"dataset_name": "wine", "model_type": "random_forest"},
    },
    "train_pytorch_mlp": {
        "func": train_pytorch_mlp,
        "description": "Train a 1-hidden-layer PyTorch MLP and return final loss and "
                       "test accuracy.",
        "example": {"dataset_name": "breast_cancer", "hidden_dim": 32, "epochs": 50},
    },
    "tune_hyperparameters": {
        "func": tune_hyperparameters,
        "description": "Run GridSearchCV or RandomizedSearchCV over an SVC (kernel SVM) or a "
                       "decision_tree, returning the best hyperparameters and CV accuracy.",
        "example": {"dataset_name": "wine", "model_type": "svc", "search_type": "grid"},
    },
    "feature_selection": {
        "func": feature_selection,
        "description": "Apply PCA and/or Sequential Feature Selection and compare the reduced "
                       "representation against the full-feature baseline accuracy.",
        "example": {"dataset_name": "breast_cancer", "method": "both", "n_components": 5},
    },
    "train_deep_classifier": {
        "func": train_deep_classifier,
        "description": "Train a deep PyTorch MLP with Dropout, BatchNorm and a learning-rate "
                       "scheduler (cosine, step, plateau, none). Reports the generalisation gap.",
        "example": {"dataset_name": "breast_cancer", "hidden_dims": [64, 32],
                    "dropout": 0.3, "scheduler": "cosine"},
    },
}

# Mapping kept for backwards compatibility with the assignment brief's naming.
AVAILABLE_TOOLS: Dict[str, Callable[..., str]] = {
    name: spec["func"] for name, spec in TOOL_SPECS.items()
}

# Small local models frequently rename arguments. Rather than failing the whole step we
# normalise the well-known synonyms and record the rewrite in the observation.
ARG_ALIASES: Dict[str, str] = {
    "dataset": "dataset_name",
    "data": "dataset_name",
    "data_set": "dataset_name",
    "name": "dataset_name",
    "model": "model_type",
    "classifier": "model_type",
    "algorithm": "model_type",
    "estimator": "model_type",
    "learning_rate": "lr",
    "num_epochs": "epochs",
    "n_epochs": "epochs",
    "hidden": "hidden_dim",
    "hidden_size": "hidden_dim",
    "hidden_layers": "hidden_dims",
    "layers": "hidden_dims",
    "search": "search_type",
    "n_folds": "cv",
    "cv_folds": "cv",
    "components": "n_components",
    "n_features": "n_components",
    "dropout_rate": "dropout",
    "lr_scheduler": "scheduler",
}


def _signature_of(tool_name: str) -> Dict[str, Any]:
    """Return {param_name: default} for a tool; REQUIRED marks parameters with no default."""
    sig = inspect.signature(TOOL_SPECS[tool_name]["func"])
    out: Dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        out[pname] = "REQUIRED" if param.default is inspect.Parameter.empty else param.default
    return out


def describe_tools() -> str:
    """Render the tool catalogue exactly as it is injected into the ReAct system prompt."""
    lines: List[str] = []
    for i, (name, spec) in enumerate(TOOL_SPECS.items(), start=1):
        params = []
        for pname, default in _signature_of(name).items():
            params.append(pname if default == "REQUIRED" else f"{pname}={json.dumps(default)}")
        lines.append(f"{i}. {name}({', '.join(params)})")
        lines.append(f"   {spec['description']}")
        lines.append(f"   Example Action Input: {json.dumps(spec['example'])}")
    return "\n".join(lines)


def call_tool(tool_name: str, kwargs: Dict[str, Any] | None = None) -> str:
    """Validate and dispatch a tool call coming from the LLM.

    This is the hallucination firewall. Every failure mode below returns a structured
    error envelope instead of raising, so the ReAct loop can read `error_type`, repair
    its Action Input and try again rather than crashing the run.
    """
    kwargs = dict(kwargs or {})

    if tool_name not in TOOL_SPECS:
        return _err(
            "unknown_tool",
            f"There is no tool named {tool_name!r}.",
            f"Available tools are exactly: {list(TOOL_SPECS)}.",
        )

    signature = _signature_of(tool_name)

    # 1. Normalise synonym argument names.
    renamed: Dict[str, str] = {}
    for key in list(kwargs):
        if key not in signature and key in ARG_ALIASES and ARG_ALIASES[key] in signature:
            kwargs[ARG_ALIASES[key]] = kwargs.pop(key)
            renamed[key] = ARG_ALIASES[key]

    # 2. Reject arguments that are still unrecognised.
    unexpected = [k for k in kwargs if k not in signature]
    if unexpected:
        return _err(
            "unexpected_argument",
            f"{tool_name} received unknown argument(s): {unexpected}.",
            f"Valid parameters for {tool_name} are {list(signature)}. "
            f"Example Action Input: {json.dumps(TOOL_SPECS[tool_name]['example'])}",
            {k: v for k, v in kwargs.items() if k in signature}
            or TOOL_SPECS[tool_name]["example"],
        )

    # 3. Reject calls that omit a required argument.
    missing = [k for k, d in signature.items() if d == "REQUIRED" and k not in kwargs]
    if missing:
        return _err(
            "missing_argument",
            f"{tool_name} is missing required argument(s): {missing}.",
            f"Example Action Input: {json.dumps(TOOL_SPECS[tool_name]['example'])}",
            TOOL_SPECS[tool_name]["example"],
        )

    # 4. Execute, converting any unexpected crash into a recoverable observation.
    try:
        result = TOOL_SPECS[tool_name]["func"](**kwargs)
    except TypeError as exc:
        return _err(
            "invalid_parameter",
            f"TypeError while calling {tool_name}: {exc}",
            f"Check argument types. Valid parameters: {list(signature)}.",
            TOOL_SPECS[tool_name]["example"],
        )
    except ValueError as exc:
        text = str(exc)
        kind = "shape_mismatch" if any(
            token in text.lower() for token in ("shape", "dimension", "n_components", "samples")
        ) else "invalid_parameter"
        return _err(kind, f"ValueError while calling {tool_name}: {text}",
                    "Adjust the offending parameter and retry.")
    except Exception as exc:  # noqa: BLE001 - deliberately broad; the agent must not die
        return _err(
            "tool_exception",
            f"{type(exc).__name__} while calling {tool_name}: {exc}",
            "Retry with the documented example arguments for this tool.",
            TOOL_SPECS[tool_name]["example"],
        )

    # Surface silent renames so the execution trace stays auditable.
    if renamed:
        try:
            payload = json.loads(result)
            payload["_normalised_arguments"] = renamed
            return json.dumps(payload)
        except json.JSONDecodeError:
            return result
    return result


if __name__ == "__main__":
    # Offline smoke test: exercises every tool without needing Ollama or a network.
    print(describe_tools())
    print("\n--- smoke test ---")
    for tool, args in [
        ("load_dataset_summary", {"dataset_name": "iris"}),
        ("train_sklearn_model", {"dataset_name": "wine", "model_type": "random_forest"}),
        ("train_pytorch_mlp", {"dataset_name": "iris", "epochs": 30}),
        ("tune_hyperparameters", {"dataset_name": "iris", "model_type": "svc",
                                  "search_type": "random", "n_iter": 8}),
        ("feature_selection", {"dataset_name": "wine", "method": "pca", "n_components": 3}),
        ("train_deep_classifier", {"dataset_name": "iris", "hidden_dims": [32, 16],
                                   "epochs": 30}),
    ]:
        print(f"\n>>> {tool}({args})\n{call_tool(tool, args)}")
