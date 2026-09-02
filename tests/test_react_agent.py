"""Offline verification of the ReAct controller.

A ScriptedLLM stands in for Ollama, so the parser, the tool dispatcher, the
self-correction engine and the loop-breaker are all tested deterministically without a
model server. This is what makes the Task 3 self-healing behaviour reviewable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react_agent import (  # noqa: E402
    ReActAgent,
    _build_repair_hint,
    parse_action_input,
    parse_llm_output,
    save_run,
)


class ScriptedLLM:
    """Replays a fixed list of model turns and records the prompts it was given."""

    def __init__(self, turns: List[str]):
        self.turns = list(turns)
        self.prompts: List[str] = []

    def __call__(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        self.prompts.append(prompt)
        if not self.turns:
            return "Thought: done.\nFinal Answer: out of scripted turns.", {}
        return self.turns.pop(0), {"eval_count": 42, "tokens_per_second": 25.0}


def agent(turns: List[str], **kwargs) -> Tuple[ReActAgent, ScriptedLLM]:
    llm = ScriptedLLM(turns)
    return ReActAgent(llm=llm, model="scripted", verbose=False, **kwargs), llm


# --------------------------------------------------------------------------------------
# Action Input parsing
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"dataset_name": "iris"}', {"dataset_name": "iris"}),
        ('```json\n{"dataset_name": "iris"}\n```', {"dataset_name": "iris"}),
        ("{'dataset_name': 'iris', 'epochs': 50}", {"dataset_name": "iris", "epochs": 50}),
        ('{"dataset_name": "iris"}  then I will compare', {"dataset_name": "iris"}),
        ('dataset_name="iris", model_type="random_forest"',
         {"dataset_name": "iris", "model_type": "random_forest"}),
        ('{"hidden_dims": [64, 32], "dropout": 0.3}', {"hidden_dims": [64, 32], "dropout": 0.3}),
        ('{"note": "a } brace inside a string"}', {"note": "a } brace inside a string"}),
    ],
)
def test_parse_action_input_tolerates_messy_model_output(raw, expected):
    kwargs, err = parse_action_input(raw)
    assert err is None
    assert kwargs == expected


def test_parse_action_input_reports_unparseable_text():
    kwargs, err = parse_action_input("just some prose with no arguments at all")
    assert kwargs is None
    assert "Could not parse" in err


# --------------------------------------------------------------------------------------
# Turn parsing
# --------------------------------------------------------------------------------------
def test_parse_clean_action_turn():
    out = parse_llm_output(
        'Thought: I should inspect the data first.\n'
        'Action: load_dataset_summary\n'
        'Action Input: {"dataset_name": "iris"}'
    )
    assert out["thought"].startswith("I should inspect")
    assert out["action"] == "load_dataset_summary"
    assert out["action_input"] == {"dataset_name": "iris"}
    assert out["parse_error"] is None


def test_parse_final_answer_turn():
    out = parse_llm_output(
        "Thought: I have gathered all necessary experimental data.\n"
        "Final Answer: Random Forest wins at 0.96 accuracy."
    )
    assert out["final_answer"] == "Random Forest wins at 0.96 accuracy."
    assert out["action"] is None


def test_parse_function_call_style_action():
    out = parse_llm_output(
        'Thought: try the tree.\n'
        'Action: train_sklearn_model(dataset_name="wine", model_type="decision_tree")'
    )
    assert out["action"] == "train_sklearn_model"
    assert out["action_input"] == {"dataset_name": "wine", "model_type": "decision_tree"}


def test_parse_bold_markdown_headers():
    out = parse_llm_output(
        'Thought: check it.\n**Action:** load_dataset_summary\n'
        '**Action Input:** {"dataset_name": "wine"}'
    )
    assert out["action"] == "load_dataset_summary"
    assert out["action_input"] == {"dataset_name": "wine"}


def test_parse_turn_with_no_protocol_at_all_is_flagged():
    out = parse_llm_output("Sure! Let me help you with that machine learning task.")
    assert out["parse_error"] is not None
    assert out["action"] is None


# --------------------------------------------------------------------------------------
# End-to-end loop behaviour
# --------------------------------------------------------------------------------------
def test_single_tool_run_completes():
    a, _ = agent([
        'Thought: inspect iris.\nAction: load_dataset_summary\n'
        'Action Input: {"dataset_name": "iris"}',
        "Thought: I have gathered all necessary experimental data.\n"
        "Final Answer: iris has 150 samples and 4 features.",
    ])
    record = a.run("Describe the iris dataset.")
    assert record.completed is True
    assert len(record.steps) == 2
    assert record.steps[0].event == "action"
    assert json.loads(record.steps[0].observation)["n_samples"] == 150
    assert record.steps[1].event == "final_answer"


def test_multi_tool_run_chains_three_tools():
    a, _ = agent([
        'Thought: summary first.\nAction: load_dataset_summary\n'
        'Action Input: {"dataset_name": "wine"}',
        'Thought: now a forest.\nAction: train_sklearn_model\n'
        'Action Input: {"dataset_name": "wine", "model_type": "random_forest"}',
        'Thought: now a neural net.\nAction: train_pytorch_mlp\n'
        'Action Input: {"dataset_name": "wine", "epochs": 30}',
        "Thought: I have gathered all necessary experimental data.\nFinal Answer: done.",
    ])
    record = a.run("Compare a forest and an MLP on wine.")
    assert record.completed is True
    assert [s.action for s in record.steps[:3]] == [
        "load_dataset_summary", "train_sklearn_model", "train_pytorch_mlp"]
    assert all(json.loads(s.observation)["status"] == "ok" for s in record.steps[:3])


def test_agent_self_corrects_a_shape_mismatch():
    """The headline Task 3 behaviour: bad n_components -> repaired -> success."""
    a, llm = agent([
        # 20 components requested from a 4-feature dataset.
        'Thought: reduce iris.\nAction: feature_selection\n'
        'Action Input: {"dataset_name": "iris", "method": "pca", "n_components": 20}',
        # After reading the hint the model retries with a legal value.
        'Thought: iris only has 4 features, so I will use 2.\nAction: feature_selection\n'
        'Action Input: {"dataset_name": "iris", "method": "pca", "n_components": 2}',
        "Thought: I have gathered all necessary experimental data.\n"
        "Final Answer: two components retain most of the variance.",
    ])
    record = a.run("Run PCA on iris.")

    assert record.repairs == 1
    assert record.completed is True
    assert record.steps[0].event == "repair"
    assert json.loads(record.steps[0].observation)["error_type"] == "shape_mismatch"
    # The corrective instruction must actually have reached the model.
    assert "[SELF-CORRECTION 1]" in llm.prompts[1]
    assert '"n_components": 4' in llm.prompts[1]
    assert json.loads(record.steps[1].observation)["status"] == "ok"


def test_agent_self_corrects_a_hallucinated_argument_name():
    a, llm = agent([
        'Thought: train a forest.\nAction: train_sklearn_model\n'
        'Action Input: {"dataset_name": "iris", "n_estimators": 500}',
        'Thought: I must supply model_type.\nAction: train_sklearn_model\n'
        'Action Input: {"dataset_name": "iris", "model_type": "random_forest"}',
        "Thought: I have gathered all necessary experimental data.\nFinal Answer: 0.93.",
    ])
    record = a.run("Train a random forest on iris.")
    assert record.repairs == 1
    assert json.loads(record.steps[0].observation)["error_type"] == "unexpected_argument"
    assert "[SELF-CORRECTION 1]" in llm.prompts[1]
    assert record.completed is True


def test_repair_budget_stops_an_endlessly_failing_call():
    bad = ('Thought: again.\nAction: load_dataset_summary\n'
           'Action Input: {"dataset_name": "titanic"}')
    a, llm = agent([bad] * 6, max_repairs_per_signature=2, max_iterations=6)
    record = a.run("Summarise titanic.")
    assert record.completed is False
    assert record.repairs == 6
    # Once the budget is spent the agent stops offering a retry_with and says so.
    assert "has now failed 3 times" in llm.prompts[-1]


def test_identical_successful_call_is_served_from_cache():
    same = ('Thought: summarise.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, llm = agent([
        same, same,
        "Thought: I have gathered all necessary experimental data.\nFinal Answer: 150 rows.",
    ])
    record = a.run("Describe iris.")
    assert record.steps[0].event == "action"
    assert record.steps[1].event == "cached"
    assert "already executed earlier" in llm.prompts[2]
    assert record.completed is True


def test_protocol_violation_triggers_a_reformat_nudge_without_running_a_tool():
    a, llm = agent([
        "Sure, I would be happy to help you analyse that dataset!",
        'Thought: sorry.\nAction: load_dataset_summary\n'
        'Action Input: {"dataset_name": "iris"}',
        "Thought: I have gathered all necessary experimental data.\nFinal Answer: ok.",
    ])
    record = a.run("Describe iris.")
    assert record.parse_errors == 1
    assert record.steps[0].event == "parse_error"
    assert record.steps[0].action is None
    assert "Reply with EXACTLY these three lines" in llm.prompts[1]
    assert record.completed is True


def test_hallucinated_tool_name_is_survivable():
    a, _ = agent([
        'Thought: use xgboost.\nAction: train_xgboost\nAction Input: {"dataset_name": "iris"}',
        'Thought: not available, use the forest.\nAction: train_sklearn_model\n'
        'Action Input: {"dataset_name": "iris", "model_type": "random_forest"}',
        "Thought: I have gathered all necessary experimental data.\nFinal Answer: ok.",
    ])
    record = a.run("Train the best model on iris.")
    assert json.loads(record.steps[0].observation)["error_type"] == "unknown_tool"
    assert record.completed is True


def test_iteration_budget_is_enforced():
    turn = ('Thought: loop.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, _ = agent([turn] * 10, max_iterations=3)
    record = a.run("Loop forever.")
    assert record.completed is False
    assert len(record.steps) == 3


# --------------------------------------------------------------------------------------
# Repair-hint construction and telemetry
# --------------------------------------------------------------------------------------
def test_repair_hint_includes_strategy_hint_and_retry_arguments():
    envelope = json.dumps({
        "status": "error", "error_type": "nan_loss",
        "message": "Loss became nan at epoch 3.",
        "hint": "The learning rate is too high.",
        "retry_with": {"dataset_name": "iris", "lr": 0.001},
    })
    hint = _build_repair_hint(envelope, 1)
    assert "[SELF-CORRECTION 1]" in hint
    assert "learning rate ten times smaller" in hint
    assert '"lr": 0.001' in hint


def test_repair_hint_is_none_for_successful_observations():
    assert _build_repair_hint(json.dumps({"status": "ok", "dataset": "iris"}), 1) is None


def test_latency_summary_and_log_files_are_written(tmp_path):
    a, _ = agent([
        'Thought: go.\nAction: load_dataset_summary\nAction Input: {"dataset_name": "iris"}',
        "Thought: I have gathered all necessary experimental data.\nFinal Answer: done.",
    ])
    record = a.run("Describe iris.")
    summary = record.latency_summary()
    assert summary["llm_calls"] == 2
    assert summary["tokens_per_second_mean"] == 25.0

    paths = save_run(record, log_dir=tmp_path, label="unit")
    assert paths["log"].exists() and paths["json"].exists()
    saved = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert saved["completed"] is True
    assert saved["task"] == "Describe iris."
    assert "FINAL ANSWER" in paths["log"].read_text(encoding="utf-8")
