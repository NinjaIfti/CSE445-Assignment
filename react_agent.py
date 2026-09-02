"""
react_agent.py -- Autonomous ReAct controller driving a local quantized LLM via Ollama.

The controller implements the Reason + Act loop from first principles (no LangChain, no
external agent framework) so that every stage is inspectable:

    Thought -> Action -> Action Input -> Observation -> ... -> Final Answer

Three properties matter for the assignment and are implemented explicitly here.

  1. Robust parsing (Task 1). A 3B parameter model does not emit clean protocol text. The
     parser tolerates markdown fences, function-call syntax, single-quoted dicts, trailing
     prose and duplicated headers before it gives up and asks for a reformat.

  2. Self-correction (Task 3). Tool failures come back as structured envelopes carrying an
     `error_type` and often a `retry_with` dictionary. `_build_repair_hint` turns those
     into an explicit corrective instruction appended to the scratchpad, so the model
     repairs the call rather than repeating it. Repairs are budgeted per failure signature
     so a persistently broken call cannot spin forever.

  3. Observability. Every step records LLM latency plus Ollama's own token counters, and
     the whole run is written to logs/ as both a human-readable trace and a JSON record.

Usage
    python react_agent.py "Compare a random forest and a deep MLP on breast_cancer."
    python react_agent.py --task-file tasks/benchmark.txt --model llama3.2:3b -v
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from ml_tools import TOOL_SPECS, call_tool, describe_tools

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
LOG_DIR = Path(__file__).resolve().parent / "logs"

SYSTEM_PROMPT_TEMPLATE = """You are an expert Autonomous Machine Learning Assistant.
You solve machine learning problems by thinking step-by-step and invoking external tools.
You never invent numerical results: every number you report must come from an Observation.

You have access to the following tools:
{tool_catalogue}

To use a tool, you MUST strictly use this format:
Thought: describe your reasoning about what to do next.
Action: the exact tool name from the list above
Action Input: a single-line JSON object of arguments, e.g. {{"dataset_name": "iris"}}

Then STOP and wait. The system will reply with a line beginning "Observation:".

Rules:
- Emit exactly one Action per turn. Never write the Observation yourself.
- Action Input must be valid JSON on one line. Use double quotes. No trailing commas.
- If an Observation has "status": "error", read "error_type" and "hint", then retry the
  same tool with corrected arguments. If the Observation contains "retry_with", use those
  arguments verbatim.
- Do not repeat a tool call that already succeeded; reuse the number from its Observation.

When you have gathered every result you need, respond in this format instead:
Thought: I have gathered all necessary experimental data.
Final Answer: your complete answer, including any comparison table the user asked for.

Begin!
"""


# --------------------------------------------------------------------------------------
# Data records
# --------------------------------------------------------------------------------------
@dataclass
class StepRecord:
    """One full Thought/Action/Observation cycle, including timing telemetry."""

    step: int
    raw_output: str = ""
    thought: str = ""
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None
    event: str = "action"          # action | final_answer | parse_error | repair | cached
    llm_seconds: float = 0.0
    eval_count: int = 0
    tokens_per_second: float = 0.0
    tool_seconds: float = 0.0


@dataclass
class RunRecord:
    """Everything needed to reproduce and grade a single agent run."""

    task: str
    model: str
    started_at: str
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    completed: bool = False
    repairs: int = 0
    parse_errors: int = 0
    total_seconds: float = 0.0

    def latency_summary(self) -> Dict[str, float]:
        # Every recorded step corresponds to exactly one LLM call, so the step count is
        # the call count. Do not filter on a non-zero duration: a fast call can legitimately
        # round to 0.0 and would silently vanish from the statistics.
        lat = [s.llm_seconds for s in self.steps]
        tps = [s.tokens_per_second for s in self.steps if s.tokens_per_second > 0]
        if not lat:
            return {}
        ordered = sorted(lat)
        return {
            "llm_calls": len(lat),
            "llm_seconds_total": round(sum(lat), 3),
            "llm_seconds_mean": round(sum(lat) / len(lat), 3),
            "llm_seconds_p50": round(ordered[len(ordered) // 2], 3),
            "llm_seconds_max": round(max(lat), 3),
            "tokens_per_second_mean": round(sum(tps) / len(tps), 2) if tps else 0.0,
            "tool_seconds_total": round(sum(s.tool_seconds for s in self.steps), 3),
        }


# --------------------------------------------------------------------------------------
# Local LLM transport
# --------------------------------------------------------------------------------------
class OllamaClient:
    """Thin client over the Ollama /api/generate endpoint served inside WSL2."""

    def __init__(self, model: str = MODEL_NAME, url: str = OLLAMA_URL,
                 temperature: float = 0.1, timeout: int = 300):
        self.model, self.url, self.temperature, self.timeout = model, url, temperature, timeout

    def __call__(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                # Stopping at "Observation:" is what prevents the model from hallucinating
                # tool output and continuing the conversation on its own.
                "stop": ["Observation:", "\nObservation"],
                "num_predict": 512,
            },
        }
        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.url}. Inside WSL run 'ollama serve &' "
                f"and confirm with 'curl {self.url.replace('/api/generate', '')}'. ({exc})"
            ) from exc
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text[:400]}")

        body = response.json()
        eval_count = int(body.get("eval_count", 0) or 0)
        eval_ns = int(body.get("eval_duration", 0) or 0)
        meta = {
            "eval_count": eval_count,
            "prompt_eval_count": int(body.get("prompt_eval_count", 0) or 0),
            "tokens_per_second": round(eval_count / (eval_ns / 1e9), 2) if eval_ns else 0.0,
            "total_seconds": round(int(body.get("total_duration", 0) or 0) / 1e9, 3),
        }
        return body.get("response", ""), meta

    def health(self) -> Dict[str, Any]:
        """Check the daemon is up and the requested model has actually been pulled."""
        base = self.url.split("/api/")[0]
        try:
            tags = requests.get(f"{base}/api/tags", timeout=10).json()
        except Exception as exc:  # noqa: BLE001
            return {"reachable": False, "error": str(exc), "url": base}
        names = [m.get("name", "") for m in tags.get("models", [])]
        return {
            "reachable": True,
            "url": base,
            "models": names,
            "requested_model_present": any(n.split(":")[0] == self.model.split(":")[0]
                                           for n in names),
        }


# --------------------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------------------
_ACTION_RE = re.compile(r"Action\s*:?\s*\**\s*([A-Za-z0-9_]+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*:?\s*\**\s*(.+)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.+?)(?=\n\s*(?:Action|Final Answer)|\Z)",
                         re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:?\s*\**\s*(.*)", re.IGNORECASE | re.DOTALL)
_CALL_RE = re.compile(r"([A-Za-z0-9_]+)\s*\((.*)\)\s*$", re.DOTALL)


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} block, ignoring braces inside string literals."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_action_input(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Coerce whatever the model produced into a kwargs dict.

    Returns (kwargs, error_message). Handles, in order: markdown fences, a balanced JSON
    object, a Python-literal dict with single quotes, and bare `key=value` pairs.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json|python)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = text.split("```")[0].strip()
    if not text:
        return None, "Action Input was empty."

    candidate = _extract_first_json_object(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError:
            # Single quotes / Python True / trailing commas are the usual culprits.
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return {str(k): v for k, v in parsed.items()}, None
            except (ValueError, SyntaxError):
                pass

    # Fallback: `dataset_name="iris", epochs=50` emitted without any braces.
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\n]+)", text.split("\n")[0])
    if pairs:
        kwargs: Dict[str, Any] = {}
        for key, value in pairs:
            value = value.strip().strip("\"'")
            try:
                kwargs[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                kwargs[key] = value
        return kwargs, None

    return None, f"Could not parse Action Input as JSON: {text[:160]!r}"


def parse_llm_output(text: str) -> Dict[str, Any]:
    """Split one model turn into thought / action / action_input / final_answer."""
    result: Dict[str, Any] = {
        "thought": "", "action": None, "action_input": None,
        "final_answer": None, "parse_error": None,
    }

    thought = _THOUGHT_RE.search(text)
    if thought:
        result["thought"] = thought.group(1).strip()

    final = _FINAL_RE.search(text)
    if final and final.group(1).strip():
        result["final_answer"] = final.group(1).strip()
        return result

    action = _ACTION_RE.search(text)
    if not action:
        result["parse_error"] = "No 'Action:' line and no 'Final Answer:' line was found."
        return result

    tool_name = action.group(1).strip()
    action_input_match = _ACTION_INPUT_RE.search(text)

    if not action_input_match:
        # `Action: train_sklearn_model(dataset_name="iris", model_type="decision_tree")`
        tail = text[action.start():].split("\n")[0]
        call = _CALL_RE.search(tail)
        if call:
            tool_name = call.group(1).strip()
            kwargs, err = parse_action_input("{" + call.group(2) + "}") if "=" not in call.group(2) \
                else parse_action_input(call.group(2))
            result["action"], result["action_input"] = tool_name, kwargs
            result["parse_error"] = err
            return result
        result["action"] = tool_name
        result["parse_error"] = "Found 'Action:' but no 'Action Input:' line."
        return result

    kwargs, err = parse_action_input(action_input_match.group(1))
    result["action"], result["action_input"], result["parse_error"] = tool_name, kwargs, err
    return result


# --------------------------------------------------------------------------------------
# Self-correction (Task 3)
# --------------------------------------------------------------------------------------
# Error type -> the corrective instruction appended after the failing Observation.
REPAIR_STRATEGIES: Dict[str, str] = {
    "unknown_tool": "That tool does not exist. Choose a tool name verbatim from the tool "
                    "list and repeat the Action.",
    "unknown_dataset": "The dataset name was wrong. Re-issue the same Action using one of "
                       "the exact dataset names listed in the hint.",
    "unknown_model": "That model is not supported by this tool. Pick a supported model_type "
                     "from the hint, or switch to the tool that does support it.",
    "unexpected_argument": "You passed an argument this tool does not accept. Re-issue the "
                           "Action using only the valid parameter names from the hint.",
    "missing_argument": "A required argument was missing. Re-issue the Action with every "
                        "required parameter supplied.",
    "invalid_parameter": "One argument had an invalid type or value. Correct just that "
                         "argument and re-issue the Action.",
    "shape_mismatch": "The requested dimensionality is incompatible with the data. Reduce "
                      "it to fit within the number of available features and retry.",
    "nan_loss": "Training diverged (loss became NaN or infinite). Retry the same Action "
                "with a learning rate ten times smaller.",
    "search_failed": "The hyperparameter search could not run with those settings. Retry "
                     "with the default scoring and search space.",
    "tool_exception": "The tool crashed. Retry once with the documented example arguments; "
                      "if it fails again, choose a different tool.",
}

_PARSE_REMINDER = (
    "Your last message did not follow the required protocol ({reason}).\n"
    "Reply with EXACTLY these three lines and nothing else:\n"
    "Thought: <one sentence>\n"
    "Action: <one tool name from the list>\n"
    'Action Input: {{"dataset_name": "iris"}}\n'
    "Or, if you already have every result you need, reply with 'Final Answer:' followed "
    "by your answer."
)


def _build_repair_hint(observation_json: str, attempts: int) -> Optional[str]:
    """Turn a failed tool Observation into an explicit corrective instruction."""
    try:
        payload = json.loads(observation_json)
    except json.JSONDecodeError:
        return None
    if payload.get("status") != "error":
        return None

    error_type = payload.get("error_type", "tool_exception")
    parts = [f"[SELF-CORRECTION {attempts}] The call failed with error_type="
             f"'{error_type}'."]
    parts.append(REPAIR_STRATEGIES.get(error_type, REPAIR_STRATEGIES["tool_exception"]))
    if payload.get("hint"):
        parts.append(f"Hint: {payload['hint']}")
    if payload.get("retry_with"):
        parts.append("Use exactly this Action Input: " + json.dumps(payload["retry_with"]))
    return " ".join(parts)


# --------------------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------------------
class ReActAgent:
    """Reason+Act controller with tool dispatch, self-correction and run logging."""

    def __init__(
        self,
        llm: Optional[Callable[[str], Tuple[str, Dict[str, Any]]]] = None,
        model: str = MODEL_NAME,
        max_iterations: int = 12,
        max_repairs_per_signature: int = 3,
        verbose: bool = True,
    ):
        self.llm = llm or OllamaClient(model=model)
        self.model = model
        self.max_iterations = max_iterations
        self.max_repairs_per_signature = max_repairs_per_signature
        self.verbose = verbose
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tool_catalogue=describe_tools())

    # -- output helpers ----------------------------------------------------------------
    def _say(self, text: str = "") -> None:
        if self.verbose:
            print(text, flush=True)

    # -- main loop ---------------------------------------------------------------------
    def run(self, task: str) -> RunRecord:
        record = RunRecord(task=task, model=self.model,
                           started_at=datetime.now().isoformat(timespec="seconds"))
        prompt = f"{self.system_prompt}\nUser Query: {task}\n"

        # Memoise successful calls: identical repeated Actions are the single most common
        # failure mode of small models and would otherwise burn the whole iteration budget.
        cache: Dict[str, str] = {}
        repair_counts: Dict[str, int] = {}

        self._say("=" * 78)
        self._say(f"USER QUERY: {task}")
        self._say(f"MODEL: {self.model}   MAX ITERATIONS: {self.max_iterations}")
        self._say("=" * 78)

        run_start = time.perf_counter()
        for step_no in range(1, self.max_iterations + 1):
            step = StepRecord(step=step_no)
            self._say(f"\n--- Step {step_no} ---")

            call_start = time.perf_counter()
            llm_output, meta = self.llm(prompt)
            step.llm_seconds = round(time.perf_counter() - call_start, 3)
            step.eval_count = int(meta.get("eval_count", 0))
            step.tokens_per_second = float(meta.get("tokens_per_second", 0.0))
            step.raw_output = llm_output
            self._say(llm_output.strip())
            self._say(f"[latency: {step.llm_seconds}s | {step.eval_count} tokens "
                      f"@ {step.tokens_per_second} tok/s]")

            prompt += llm_output
            parsed = parse_llm_output(llm_output)
            step.thought = parsed["thought"]

            # 1. Terminal state.
            if parsed["final_answer"]:
                step.event = "final_answer"
                step.final_answer = parsed["final_answer"]
                record.steps.append(step)
                record.final_answer = parsed["final_answer"]
                record.completed = True
                self._say("\n>>> Task completed successfully.")
                break

            # 2. Protocol violation -> reformat instruction, no tool executed.
            if parsed["parse_error"] or not parsed["action"] or parsed["action_input"] is None:
                reason = parsed["parse_error"] or "no Action Input could be recovered"
                step.event = "parse_error"
                step.action = parsed["action"]
                step.observation = f"Observation: {reason}"
                record.parse_errors += 1
                nudge = "\nObservation: " + _PARSE_REMINDER.format(reason=reason) + "\n"
                prompt += nudge
                record.steps.append(step)
                self._say(nudge.strip())
                continue

            tool_name = parsed["action"]
            kwargs = parsed["action_input"]
            step.action, step.action_input = tool_name, kwargs

            # 3. Loop-breaker: serve an identical earlier call from cache.
            signature = f"{tool_name}::{json.dumps(kwargs, sort_keys=True, default=str)}"
            if signature in cache:
                step.event = "cached"
                step.observation = cache[signature]
                observation = (
                    f"\nObservation: {cache[signature]}\n"
                    "[NOTE] This exact call was already executed earlier in this run. Do not "
                    "repeat it again. Either call a different tool or give your Final Answer.\n"
                )
                prompt += observation
                record.steps.append(step)
                self._say(observation.strip())
                continue

            # 4. Execute the tool.
            tool_start = time.perf_counter()
            result = call_tool(tool_name, kwargs)
            step.tool_seconds = round(time.perf_counter() - tool_start, 3)
            step.observation = result
            self._say(f"Observation: {result}")

            repair_hint = None
            try:
                is_error = json.loads(result).get("status") == "error"
            except json.JSONDecodeError:
                is_error = False

            if is_error:
                repair_counts[signature] = repair_counts.get(signature, 0) + 1
                attempts = repair_counts[signature]
                record.repairs += 1
                step.event = "repair"
                if attempts <= self.max_repairs_per_signature:
                    repair_hint = _build_repair_hint(result, attempts)
                else:
                    repair_hint = (
                        f"[SELF-CORRECTION] This call has now failed {attempts} times. Stop "
                        "retrying it. Use a different tool or different arguments, or give "
                        "your Final Answer using the results you already have."
                    )
            else:
                cache[signature] = result

            observation = f"\nObservation: {result}\n"
            if repair_hint:
                observation += repair_hint + "\n"
                self._say(repair_hint)
            prompt += observation
            record.steps.append(step)
        else:
            self._say(f"\n>>> Iteration budget ({self.max_iterations}) exhausted "
                      "without a Final Answer.")

        record.total_seconds = round(time.perf_counter() - run_start, 3)
        return record


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
def save_run(record: RunRecord, log_dir: Path = LOG_DIR, label: str = "trace") -> Dict[str, Path]:
    """Write a human-readable transcript and a machine-readable JSON record."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = log_dir / f"{label}_{stamp}.log"
    json_path = log_dir / f"{label}_{stamp}.json"

    lines = [
        "=" * 78,
        f"CSE445 Assignment 3 -- agent execution trace",
        f"Task      : {record.task}",
        f"Model     : {record.model}",
        f"Started   : {record.started_at}",
        f"Completed : {record.completed}   Steps: {len(record.steps)}   "
        f"Repairs: {record.repairs}   Parse errors: {record.parse_errors}",
        "=" * 78,
    ]
    for step in record.steps:
        lines += [f"\n--- Step {step.step} [{step.event}] ---", step.raw_output.strip()]
        if step.action:
            lines.append(f"Action: {step.action}")
            lines.append(f"Action Input: {json.dumps(step.action_input)}")
        if step.observation:
            lines.append(f"Observation: {step.observation}")
        lines.append(
            f"[latency {step.llm_seconds}s | {step.eval_count} tok "
            f"@ {step.tokens_per_second} tok/s | tool {step.tool_seconds}s]"
        )
    if record.final_answer:
        lines += ["\n" + "=" * 78, "FINAL ANSWER", "=" * 78, record.final_answer]
    lines += ["\n" + "=" * 78, "LATENCY SUMMARY",
              json.dumps(record.latency_summary(), indent=2)]

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    payload = asdict(record)
    payload["latency_summary"] = record.latency_summary()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"log": txt_path, "json": json_path}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
DEFAULT_TASK = (
    "Analyze the breast_cancer dataset, train a Random Forest and a PyTorch MLP on it, "
    "compare their accuracies, and recommend the best model for clinical screening."
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("task", nargs="?", default=None, help="natural-language task")
    parser.add_argument("--task-file", help="read the task from a file instead")
    parser.add_argument("--model", default=MODEL_NAME, help=f"Ollama model (default {MODEL_NAME})")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--label", default="trace", help="log filename prefix")
    parser.add_argument("--no-log", action="store_true", help="do not write logs/")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--health", action="store_true",
                        help="check the Ollama daemon and exit")
    args = parser.parse_args(argv)

    client = OllamaClient(model=args.model, temperature=args.temperature)

    if args.health:
        status = client.health()
        print(json.dumps(status, indent=2))
        if not status.get("reachable"):
            print("\nOllama is not reachable. Inside WSL: 'ollama serve &'", file=sys.stderr)
            return 1
        if not status.get("requested_model_present"):
            print(f"\nModel '{args.model}' not pulled. Run: ollama pull {args.model}",
                  file=sys.stderr)
            return 1
        print(f"\nOK: '{args.model}' is available. Tools registered: {list(TOOL_SPECS)}")
        return 0

    task = args.task
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8").strip()
    task = task or DEFAULT_TASK

    agent = ReActAgent(llm=client, model=args.model, max_iterations=args.max_iterations,
                       verbose=not args.quiet)
    try:
        record = agent.run(task)
    except RuntimeError as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("LATENCY SUMMARY:", json.dumps(record.latency_summary(), indent=2))
    if not args.no_log:
        paths = save_run(record, label=args.label)
        print(f"Trace written to {paths['log']}")
        print(f"JSON  written to {paths['json']}")
    return 0 if record.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
