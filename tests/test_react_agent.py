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
    audit_grounding,
    is_fabricated,
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
        self.stop_overrides: List[Any] = []

    def __call__(self, prompt: str, stop: Any = None) -> Tuple[str, Dict[str, Any]]:
        self.prompts.append(prompt)
        self.stop_overrides.append(stop)
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
    # Three consecutive failures trip the stall detector, so the loop stops early
    # rather than spending the full six iterations on a call that cannot succeed.
    assert record.repairs == 3
    # Once the repair budget is spent the agent stops offering a retry_with and says so.
    assert "has now failed 3 times" in llm.prompts[-1]
    # A run with nothing but failures still terminates through forced finalisation.
    assert record.forced_final is True


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
    a, _ = agent([turn] * 10, max_iterations=3, max_unproductive_steps=99)
    record = a.run("Loop forever.")
    # Exactly max_iterations reasoning steps, plus the single forced-final step.
    assert [s.event for s in record.steps] == ["action", "cached", "cached", "forced_final"]
    assert record.forced_final is True


# --------------------------------------------------------------------------------------
# Forced finalisation: the stall-breaker
# --------------------------------------------------------------------------------------
def test_forced_finalisation_rescues_a_stalled_run():
    """The real llama3.2:3b failure: it re-calls a tool it already ran and never concludes."""
    same = ('Thought: summarise.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, _ = agent([same, same, same, same,
                  " iris has 150 samples, 4 features and 3 balanced classes."],
                 max_iterations=8)
    record = a.run("Describe iris.")

    # One real call, then three cached repeats trip the stall detector.
    assert [s.event for s in record.steps] == [
        "action", "cached", "cached", "cached", "forced_final"]
    assert record.completed is True
    assert record.forced_final is True
    assert record.final_answer == "iris has 150 samples, 4 features and 3 balanced classes."


def test_forced_final_prompt_prefills_the_answer_opening():
    """Pre-filling 'Final Answer:' leaves the model no room to emit another Action."""
    same = ('Thought: go.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, llm = agent([same] * 4 + [" done."], max_iterations=8)
    a.run("Describe iris.")

    closing = llm.prompts[-1]
    assert closing.rstrip().endswith("Final Answer:")
    assert "Tool use is now closed" in closing
    # The observations gathered earlier are still present, so the answer stays grounded.
    assert '"n_samples": 150' in closing
    # The Observation stop sequence must be lifted, or the answer comes back empty.
    assert llm.stop_overrides[-1] == []
    assert llm.stop_overrides[0] is None


def test_forced_final_retries_when_the_model_echoes_observation_json():
    """A JSON dump is not an answer. One retry with a prose pre-fill must replace it."""
    same = ('Thought: go.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, llm = agent([same] * 4 + [
        '{"status": "ok", "n_samples": 150}',                 # lazy echo
        " iris has 150 samples across 3 balanced classes.",   # the retry
    ], max_iterations=8)
    record = a.run("Describe iris.")

    assert record.final_answer == "In plain English, iris has 150 samples across 3 balanced classes."
    # The retry pre-fills the opening of the sentence to block another JSON dump.
    assert llm.prompts[-1].rstrip().endswith("Final Answer: In plain English,")


def test_forced_final_accepts_good_prose_without_retrying():
    same = ('Thought: go.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, llm = agent([same] * 4 + [" Iris holds 150 samples over 3 classes."],
                   max_iterations=8)
    record = a.run("Describe iris.")

    assert record.final_answer == "Iris holds 150 samples over 3 classes."
    assert len(llm.prompts) == 5   # four loop turns plus one forced call, no retry


def test_forced_final_strips_trailing_protocol_text():
    """The model often bolts another Action onto its forced answer; it must be cut."""
    same = ('Thought: go.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, _ = agent([same] * 4 + [
        " iris has 150 samples.\nObservation: {\"fake\": 1}\nAction: load_dataset_summary"],
        max_iterations=8)
    record = a.run("Describe iris.")
    assert record.final_answer == "iris has 150 samples."


def test_a_productive_step_resets_the_stall_counter():
    iris = ('Thought: a.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    wine = ('Thought: b.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "wine"}')
    a, _ = agent([iris, iris, iris, wine, iris, iris,
                  "Thought: done.\nFinal Answer: both described."], max_iterations=10)
    record = a.run("Describe iris and wine.")

    # Two cached repeats, then real progress on wine, then two more: never three in a row.
    assert record.forced_final is False
    assert record.completed is True
    assert record.final_answer == "both described."


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


# --------------------------------------------------------------------------------------
# Grounding audit: refusing invented results tables
# --------------------------------------------------------------------------------------
def test_audit_accepts_numbers_taken_from_observations():
    obs = ['{"test_accuracy": 0.9561, "cv_mean_accuracy": 0.9543}']
    audit = audit_grounding("Random Forest reached 0.9561 test accuracy.", obs)
    assert audit["numbers_ungrounded"] == 0
    assert audit["grounded_ratio"] == 1.0
    assert is_fabricated(audit) is False


def test_audit_allows_the_model_to_round():
    """0.96 is a legitimate rendering of an observed 0.9561, not an invention."""
    obs = ['{"test_accuracy": 0.9561}']
    audit = audit_grounding("It scored about 0.96 accuracy.", obs)
    assert audit["numbers_ungrounded"] == 0


def test_audit_flags_an_entirely_invented_results_table():
    obs = ['{"status": "ok", "n_samples": 178}']
    table = (
        "| Wine | Random Forest | 0.94 | 0.93 |\n"
        "| Wine | Kernel SVM | 0.95 | 0.94 |\n"
        "| Breast Cancer | Deep NN | 0.96 | 0.95 |"
    )
    audit = audit_grounding(table, obs)
    assert audit["numbers_ungrounded"] == audit["numbers_claimed"]
    assert audit["grounded_ratio"] == 0.0
    assert is_fabricated(audit) is True


def test_a_single_stray_number_is_not_called_fabrication():
    obs = ['{"a": 0.9561, "b": 0.9543, "c": 0.9474}']
    audit = audit_grounding("Scores were 0.9561, 0.9543, 0.9474 and 0.1234.", obs)
    assert audit["numbers_ungrounded"] == 1
    assert is_fabricated(audit) is False


def test_fabricated_final_answer_is_rejected_and_sent_back_to_the_loop():
    """The real llama3.2:3b failure: a complete results table for experiments never run."""
    invented = ("Thought: done.\nFinal Answer:\n"
                "| Wine | RF | 0.94 | 0.93 |\n| Wine | SVM | 0.95 | 0.92 |")
    a, llm = agent([
        invented,
        'Thought: I should actually run it.\nAction: train_sklearn_model\n'
        'Action Input: {"dataset_name": "wine", "model_type": "random_forest"}',
        "Thought: done.\nFinal Answer: Random Forest reached 0.9775 CV accuracy on wine.",
    ], max_iterations=8)
    record = a.run("Benchmark wine.")

    assert record.grounding_rejections == 1
    assert record.steps[0].event == "grounding_rejected"
    assert "GROUNDING REJECTED" in llm.prompts[1]
    # After being pushed back it runs the experiment and answers from the real number.
    assert record.completed is True
    assert "0.9775" in record.final_answer
    assert record.grounding["numbers_ungrounded"] == 0


def test_grounded_final_answer_is_accepted_immediately():
    a, _ = agent([
        'Thought: run it.\nAction: train_sklearn_model\n'
        'Action Input: {"dataset_name": "wine", "model_type": "random_forest"}',
        "Thought: done.\nFinal Answer: Wine random forest CV accuracy was 0.9775.",
    ])
    record = a.run("Benchmark wine.")
    assert record.grounding_rejections == 0
    assert record.completed is True
    assert record.grounding["grounded_ratio"] == 1.0


def test_rejection_budget_prevents_an_endless_grounding_argument():
    invented = ("Thought: done.\nFinal Answer:\n"
                "| a | 0.11 | 0.22 |\n| b | 0.33 | 0.44 |")
    a, _ = agent([invented] * 6, max_iterations=8, max_grounding_rejections=1)
    record = a.run("Benchmark wine.")
    assert record.grounding_rejections == 1
    # Once the budget is spent the answer is kept, but recorded as ungrounded.
    assert record.completed is True
    assert is_fabricated(record.grounding) is True


def test_audit_accepts_a_proportion_reported_as_a_percentage():
    """0.9742 rendered as 97.42% is faithful reporting, not fabrication."""
    obs = ['{"test_accuracy": 0.9742, "final_loss": 0.013}']
    audit = audit_grounding("It reached 97.42% accuracy at loss 0.013.", obs)
    assert audit["numbers_ungrounded"] == 0


def test_audit_still_catches_a_mangled_percentage():
    """0.7268 explained variance reported as 92.68% is a genuine invention."""
    audit = audit_grounding("Variance retained was 92.68%.",
                            ['{"explained_variance_ratio": [0.7268, 0.2307]}'])
    assert audit["ungrounded_values"] == ["92.68"]


def test_forced_final_strips_a_restated_header_and_leading_punctuation():
    same = ('Thought: go.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, _ = agent([same] * 4 + [".\n\nFinal Answer: Iris holds 150 samples."],
                 max_iterations=8)
    record = a.run("Describe iris.")
    assert record.final_answer == "Iris holds 150 samples."


def test_forced_final_prompt_enumerates_the_permitted_numbers():
    """Listing the allowed values turns open generation into selection."""
    same = ('Thought: go.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, llm = agent([same] * 4 + [" iris has 150 samples."], max_iterations=8)
    a.run("Describe iris.")
    assert "The ONLY numeric values you may quote are:" in llm.prompts[-1]


def test_two_numbers_with_nothing_grounded_counts_as_fabrication():
    """A stalled run that never trained a model still reported two model accuracies."""
    audit = audit_grounding("Random Forest hit 97.4% and the MLP 96.8%.",
                            ['{"n_samples": 569, "n_features": 30}'])
    assert audit["numbers_claimed"] == 2
    assert audit["grounded_ratio"] == 0.0
    assert is_fabricated(audit) is True


def test_forced_answer_that_invents_results_is_retried_then_refused():
    """A forced answer gets no second lap of the loop, so it is verified in place."""
    same = ('Thought: go.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, _ = agent([same] * 4 + [
        " Random Forest hit 88.1% and the MLP 77.2%.",   # invented
        " Random Forest hit 66.3% and the MLP 55.4%.",   # still invented on retry
    ], max_iterations=8)
    record = a.run("Compare two models on iris.")

    assert record.grounding_rejections == 1
    assert record.completed is False           # a fabricated answer is not a result
    assert record.final_answer is None
    assert "66.3" in record.rejected_answer    # kept for audit, not presented
    assert record.steps[-1].event == "grounding_rejected"


def test_forced_answer_recovers_on_the_retry():
    same = ('Thought: go.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, _ = agent([same] * 4 + [
        " Accuracy was 88.1% and 77.2%.",              # invented
        " iris holds 150.0 samples over 3 classes.",   # grounded on retry
    ], max_iterations=8)
    record = a.run("Describe iris.")

    assert record.grounding_rejections == 1
    assert record.completed is True
    assert record.final_answer == "iris holds 150.0 samples over 3 classes."


def test_cached_note_tells_the_model_which_tools_remain_unused():
    same = ('Thought: go.\nAction: load_dataset_summary\n'
            'Action Input: {"dataset_name": "iris"}')
    a, llm = agent([same, same, "Thought: done.\nFinal Answer: iris has 150.0 rows."])
    a.run("Describe iris.")
    assert "Tools you have NOT yet used" in llm.prompts[2]
    assert "train_sklearn_model" in llm.prompts[2]
