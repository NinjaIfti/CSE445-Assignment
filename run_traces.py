"""
run_traces.py -- generate the execution logs required by the assignment deliverables.

The brief asks for "at least 3 multi-step agent reasoning traces with tool execution".
This script runs four scenarios chosen to demonstrate a different capability each:

    1. single_tool      one tool call, then a Final Answer (baseline ReAct loop, Task 1)
    2. multi_tool       three tools chained into one comparison (Task 1)
    3. self_correction  a deliberately impossible request that the agent must repair
                        after reading the structured error (Task 3)
    4. full_pipeline    tuning + feature selection + a regularised deep net (Task 2)

Each run writes logs/<scenario>_<timestamp>.log and .json, and the script finishes by
writing logs/TRACES.md indexing every run with its outcome and latency.

Usage
    python run_traces.py                       # all four scenarios
    python run_traces.py --only self_correction
    python run_traces.py --model mistral:7b
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from react_agent import OllamaClient, ReActAgent, save_run

LOG_DIR = Path(__file__).resolve().parent / "logs"

SCENARIOS: Dict[str, Dict[str, object]] = {
    "single_tool": {
        "title": "Single-tool execution",
        "demonstrates": "Baseline ReAct loop: one Thought/Action/Observation cycle, "
                        "then a Final Answer grounded in the observation.",
        "max_iterations": 6,
        "task": "How many samples, features and classes does the wine dataset have, "
                "and is it class-balanced?",
    },
    "multi_tool": {
        "title": "Multi-tool comparison",
        "demonstrates": "Autonomous chaining of three tools and a numerical comparison "
                        "of a classical model against a neural network.",
        "max_iterations": 12,
        "task": "Analyze the breast_cancer dataset, train a Random Forest and a PyTorch "
                "MLP on it, compare their accuracies, and recommend the best model for "
                "clinical screening.",
    },
    "self_correction": {
        "title": "Self-correction after a tool failure",
        "demonstrates": "The agent issues an impossible dimensionality, receives a "
                        "structured shape_mismatch error, reasons about it, and retries "
                        "with corrected parameters without human intervention.",
        "max_iterations": 12,
        "task": "Reduce the iris dataset to its 25 most informative principal components "
                "using the feature_selection tool, then report how much variance is "
                "retained and whether accuracy suffers compared with all features.",
    },
    "full_pipeline": {
        "title": "Full Task 2 pipeline",
        "demonstrates": "Hyperparameter search, dimensionality reduction and a "
                        "regularised deep network orchestrated in a single run.",
        "max_iterations": 16,
        "task": "For the wine dataset: first tune a kernel SVM with a grid search, then "
                "run PCA with 5 components to see how much accuracy is lost under "
                "compression, then train a deep classifier with hidden_dims [64, 32], "
                "dropout 0.3 and the cosine scheduler. Finish with a ranked summary of "
                "all three approaches.",
    },
}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--only", choices=sorted(SCENARIOS), action="append",
                        help="run only the named scenario (repeatable)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    client = OllamaClient(model=args.model)
    health = client.health()
    if not health.get("reachable"):
        print(f"Ollama is not reachable at {health.get('url')}.\n"
              "Inside WSL run:  ollama serve &", file=sys.stderr)
        return 1
    if not health.get("requested_model_present"):
        print(f"Model '{args.model}' is not pulled. Run: ollama pull {args.model}",
              file=sys.stderr)
        return 1

    selected = args.only or list(SCENARIOS)
    index: List[Dict[str, object]] = []

    for name in selected:
        scenario = SCENARIOS[name]
        print("\n" + "#" * 78)
        print(f"# SCENARIO: {name} -- {scenario['title']}")
        print("#" * 78)

        agent = ReActAgent(llm=client, model=args.model,
                           max_iterations=int(scenario["max_iterations"]),
                           verbose=not args.quiet)
        record = agent.run(str(scenario["task"]))
        paths = save_run(record, label=name)

        index.append({
            "scenario": name,
            "title": scenario["title"],
            "demonstrates": scenario["demonstrates"],
            "task": scenario["task"],
            "completed": record.completed,
            "steps": len(record.steps),
            "tool_calls": sum(1 for s in record.steps if s.action and s.event != "parse_error"),
            "repairs": record.repairs,
            "parse_errors": record.parse_errors,
            "seconds": record.total_seconds,
            "forced_final": record.forced_final,
            "grounding": record.grounding,
            "grounding_rejections": record.grounding_rejections,
            "rejected_answer": record.rejected_answer,
            "latency": record.latency_summary(),
            "log": paths["log"].name,
            "json": paths["json"].name,
            "final_answer": record.final_answer,
        })
        print(f"\n-> {paths['log']}")

    _write_index(index, args.model)
    completed = sum(1 for e in index if e["completed"])
    print(f"\n{completed}/{len(index)} scenarios reached a Final Answer.")
    print(f"Index written to {LOG_DIR / 'TRACES.md'}")
    return 0 if completed == len(index) else 2


def _write_index(index: List[Dict[str, object]], model: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Agent Execution Traces",
        "",
        f"Model `{model}` served by Ollama inside WSL2. "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `run_traces.py`.",
        "",
        "| Scenario | Steps | Tool calls | Self-corrections | Grounding | "
        "Completed | Wall (s) | Log |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in index:
        g = e["grounding"] or {}
        ground = (f"{g.get('numbers_claimed', 0) - g.get('numbers_ungrounded', 0)}"
                  f"/{g.get('numbers_claimed', 0)} numbers verified")
        lines.append(
            f"| {e['scenario']} | {e['steps']} | {e['tool_calls']} | {e['repairs']} "
            f"| {ground} | {'yes' if e['completed'] else 'REJECTED'} "
            f"| {e['seconds']} | [`{e['log']}`]({e['log']}) |"
        )

    for e in index:
        lines += [
            "",
            f"## {e['scenario']} — {e['title']}",
            "",
            f"**Demonstrates:** {e['demonstrates']}",
            "",
            f"**Task given to the agent:**",
            "",
            f"> {e['task']}",
            "",
            f"**Telemetry:** {json.dumps(e['latency'])}",
            "",
        ]
        if e["final_answer"]:
            lines += ["**Final Answer:**", "", "```", str(e["final_answer"]).strip(), "```", ""]
        elif e["rejected_answer"]:
            lines += [
                "**Final Answer: REJECTED.** The agent produced an answer whose numbers "
                "appear in no Observation from this run, so it was refused rather than "
                "reported. The rejected text is kept below for audit:",
                "", "```", str(e["rejected_answer"]).strip(), "```", "",
            ]
        else:
            lines += ["**Final Answer:** _not reached within the iteration budget._", ""]

    (LOG_DIR / "TRACES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
