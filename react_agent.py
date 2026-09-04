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
    forced_final: bool = False   # answer was compelled after a stall, not volunteered
    repairs: int = 0
    parse_errors: int = 0
    grounding: Optional[Dict[str, Any]] = None   # audit of the numbers in the answer
    grounding_rejections: int = 0                # invented answers sent back for rework
    rejected_answer: Optional[str] = None        # forced answer refused as ungrounded
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

    # Stopping at "Observation:" is what prevents the model from hallucinating tool output
    # and continuing the conversation on its own. It must be disabled for the forced-final
    # call, where "Observation" is the first thing the model tries to write and would
    # otherwise truncate the answer to an empty string.
    DEFAULT_STOP = ["Observation:", "\nObservation"]

    def __call__(self, prompt: str,
                 stop: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        # 512 truncated a legitimate Action Input mid-object; the tool call was lost to
        # the token budget rather than to any model error.
        options: Dict[str, Any] = {"temperature": self.temperature, "num_predict": 1024}
        stop_sequences = self.DEFAULT_STOP if stop is None else stop
        if stop_sequences:
            options["stop"] = stop_sequences

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
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
# The colon is REQUIRED and the header must start a line. With `Action\s*:?` the word
# "Action" occurring in ordinary prose hijacks the match: the model wrote
# "...I will retry the same Action with a learning rate ten times smaller" above a
# perfectly good "Action: train_deep_classifier", and the parser dispatched the tool
# "with". Anchoring to the line start is what distinguishes a header from a sentence.
_ACTION_RE = re.compile(r"^[ \t]*\**\s*Action\s*\**\s*:\s*\**\s*([A-Za-z0-9_]+)",
                        re.IGNORECASE | re.MULTILINE)
_ACTION_INPUT_RE = re.compile(r"^[ \t]*\**\s*Action\s*Input\s*\**\s*:\s*\**\s*(.+)",
                              re.IGNORECASE | re.DOTALL | re.MULTILINE)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.+?)(?=\n\s*(?:Action|Final Answer)|\Z)",
                         re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:?\s*\**\s*(.*)", re.IGNORECASE | re.DOTALL)
_CALL_RE = re.compile(r"([A-Za-z0-9_]+)\s*\((.*)\)\s*$", re.DOTALL)
_KV_KEY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*")


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

    # Decide by how the text starts, not by whether a brace appears anywhere. Given
    # `opts={"k": 1}, n=5`, a brace-first search grabs the *nested* object and silently
    # drops every other argument.
    if not text.startswith("{") and _KV_KEY_RE.match(text):
        kwargs = _parse_kv_pairs(text.split("\n")[0])
        if kwargs:
            return kwargs, None

    # An Action Input cut off mid-object is a truncated generation, not malformed JSON.
    # Saying so lets the model re-emit a shorter call instead of hunting for a syntax
    # error that is not there.
    if text.count("{") > text.count("}"):
        return None, ("Action Input was cut off mid-object (the generation hit the token "
                      f"limit): {text[-60:]!r}. Re-issue the call with fewer arguments, "
                      "relying on the tool defaults.")

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
    kwargs = _parse_kv_pairs(text.split("\n")[0])
    if kwargs:
        return kwargs, None

    return None, f"Could not parse Action Input as JSON: {text[:160]!r}"


def _parse_kv_pairs(text: str) -> Dict[str, Any]:
    """Parse `a=1, b=[64, 32], c="x"` into a dict, respecting brackets and quotes.

    A naive `([^,\\n]+)` value pattern splits on the comma *inside* a list, so
    `hidden_dims=[64, 32]` silently arrives as the string "[64". That is not a parse
    failure the agent can see -- the call succeeds against a different architecture than
    the one requested -- so the value scan tracks bracket depth and string state instead.
    """
    out: Dict[str, Any] = {}
    i, n = 0, len(text)
    while i < n:
        match = _KV_KEY_RE.match(text, i)
        if not match:
            i += 1
            continue
        key, start = match.group(1), match.end()
        j, depth, quote = start, 0, None
        while j < n:
            ch = text[j]
            if quote:
                if ch == quote and text[j - 1] != "\\":
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch in "[{(":
                depth += 1
            elif ch in "]})":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                break
            j += 1
        raw = text[start:j].strip()
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw.strip("\"'")
        i = j + 1
    return out


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


# --------------------------------------------------------------------------------------
# Grounding audit: catching invented numbers
# --------------------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"-?\d+\.\d+")


def audit_grounding(answer: str, observations: List[str]) -> Dict[str, Any]:
    """Check that every decimal number in the answer actually came from an Observation.

    Small models will happily produce a complete, well-formatted results table having run
    none of the experiments. The output looks authoritative and is entirely invented --
    which is far more damaging than a crash, because nothing about it signals failure.

    Rounding is allowed: an answer reporting 0.96 is grounded by an observation of 0.9561,
    so each claimed value is compared at its own stated precision.
    """
    observed: List[float] = []
    for obs in observations:
        observed.extend(float(m) for m in _NUMBER_RE.findall(obs))

    claimed = _NUMBER_RE.findall(answer)
    ungrounded = []
    for token in claimed:
        value = float(token)
        decimals = len(token.split(".")[1])
        # Direct match at the precision the answer chose to state.
        if any(abs(round(o, decimals) - value) < 1e-9 for o in observed):
            continue
        # Accuracies are stored as proportions but frequently reported as percentages,
        # so 97.42 is a faithful rendering of an observed 0.9742, not an invention.
        if any(abs(round(o * 100, decimals) - value) < 1e-9 for o in observed):
            continue
        ungrounded.append(token)

    return {
        "numbers_claimed": len(claimed),
        "numbers_ungrounded": len(ungrounded),
        "ungrounded_values": ungrounded[:12],
        "grounded_ratio": round(1 - len(ungrounded) / len(claimed), 3) if claimed else 1.0,
    }


def is_fabricated(audit: Dict[str, Any]) -> bool:
    """Treat an answer as fabricated only when the evidence is unambiguous.

    One stray number is a rounding artefact or an arithmetic slip, so a single unmatched
    value is tolerated. Two situations are not:

      * three or more unaccounted values with most of the answer ungrounded -- an
        invented results table;
      * *nothing* grounded at all across two or more values -- an answer reporting
        experiments that were never run. This case matters even at two numbers: a stalled
        run that never trained a model still confidently reported two model accuracies.
    """
    if audit["numbers_claimed"] >= 2 and audit["grounded_ratio"] == 0.0:
        return True
    # Note the inclusive bound. A real answer sat at exactly 0.5 -- seven invented values
    # out of fourteen -- and a strict "< 0.5" let it through as acceptable.
    return audit["numbers_ungrounded"] >= 3 and audit["grounded_ratio"] <= 0.5


_REGROUND_DEMAND = (
    "\nObservation: [GROUNDING REJECTED] Your Final Answer reported {n} numbers that "
    "appear in no Observation from this run: {values}. You did not run the experiments "
    "that would produce them, so those figures are invented and cannot be submitted.\n"
    "You have executed {done} successful tool call(s) so far. Continue the ReAct loop: "
    "issue the Action needed for the next missing experiment. Only once every required "
    "Observation is present may you write a Final Answer, and it must use those "
    "Observation values verbatim.\n"
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
        max_unproductive_steps: int = 3,
        max_grounding_rejections: int = 2,
        verbose: bool = True,
    ):
        self.llm = llm or OllamaClient(model=model)
        self.model = model
        self.max_iterations = max_iterations
        self.max_repairs_per_signature = max_repairs_per_signature
        self.max_unproductive_steps = max_unproductive_steps
        self.max_grounding_rejections = max_grounding_rejections
        self.verbose = verbose
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tool_catalogue=describe_tools())

    # -- output helpers ----------------------------------------------------------------
    def _say(self, text: str = "") -> None:
        if self.verbose:
            print(text, flush=True)

    @staticmethod
    def _evidence(record: RunRecord) -> List[str]:
        """Observations that count as evidence when auditing an answer's numbers.

        A rejected answer's own audit is recorded as that step's observation, and it
        necessarily quotes the invented values. Feeding it back would let a fabricated
        table validate itself on the next pass, so those steps are excluded.
        """
        return [s.observation for s in record.steps
                if s.observation and s.event != "grounding_rejected"]

    # -- forced termination ------------------------------------------------------------
    def _force_final_answer(self, prompt: str,
                            observations: Optional[List[str]] = None
                            ) -> Tuple[str, Dict[str, Any]]:
        """Compel a Final Answer by pre-filling the start of the model's turn.

        Small models get stuck in action loops: they re-issue a call whose result they
        already have, ignore the "you already ran this" note, and burn the whole budget
        without ever concluding. Asking more politely does not fix it.

        Instead of asking, this appends the opening of the answer itself to the prompt.
        Because /api/generate simply continues the text it is given, the model has no
        syntactic room left to emit another Action -- it can only write the answer. The
        observations already gathered are still in the scratchpad, so the answer stays
        grounded in real tool output rather than invented numbers.
        """
        observations = observations or []

        # Enumerating the numbers the model is permitted to use is far more effective than
        # telling it not to invent any. It turns an open-ended generation into a selection.
        allowed = sorted({m for obs in observations for m in _NUMBER_RE.findall(obs)},
                         key=float)
        allowed_clause = (
            f"The ONLY numeric values you may quote are: {', '.join(allowed[:40])}. "
            "You may express them as percentages, but you may not introduce any other "
            "number.\n" if allowed else ""
        )

        instruction = (
            "\n[SYSTEM] Tool use is now closed. You already have every Observation you "
            "need, and no further Action will be executed. Answer the user's question in "
            "plain English, in two to five sentences. Do NOT output JSON. Do NOT copy an "
            "Observation verbatim. Do NOT write another Action.\n"
            + allowed_clause +
            "Thought: I have gathered all necessary experimental data.\n"
        )

        text, meta = self._ask_for_answer(prompt + instruction + "Final Answer:")

        # A tired 3B model often ignores the prose instruction and simply echoes the
        # observation JSON. Detecting that is trivial, and one retry whose pre-fill starts
        # the sentence for it ("In plain English,") reliably produces real prose.
        if text.startswith("{") or text.startswith("["):
            retry, meta = self._ask_for_answer(
                prompt + instruction + "Final Answer: In plain English,"
            )
            if retry and not retry.startswith(("{", "[")):
                text = "In plain English, " + retry if not retry[0].isupper() else retry

        return text, meta

    def _ask_for_answer(self, full_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """One forced-answer generation, with the protocol scaffolding cleaned off."""
        # The "Observation:" stop sequence must be lifted here. Left on, the model's first
        # instinct after the pre-filled "Final Answer:" is to write another Observation
        # block, which trips the stop immediately and yields an empty answer.
        try:
            text, meta = self.llm(full_prompt, stop=[])
        except TypeError:
            # A caller-supplied llm that does not accept a stop override (e.g. a test stub).
            text, meta = self.llm(full_prompt)

        # Strip anything the model bolts on after wandering back into protocol text.
        for marker in ("\nObservation:", "\nAction:", "\nThought:", "[SYSTEM]"):
            if marker in text:
                text = text.split(marker)[0]
        text = text.strip()
        # With the stop sequence lifted, the model often opens with a stray "Observation:"
        # label before the answer itself. Drop the label, keep the content.
        if text.lower().startswith("observation:"):
            text = text.split(":", 1)[1].strip()
        # It also closes the pre-filled line with punctuation and then restates the header,
        # producing answers that begin with a stray full stop and a second "Final Answer:".
        if "Final Answer:" in text:
            text = text.split("Final Answer:", 1)[1]
        return text.strip().lstrip(".:,;- \n\t").strip(), meta

    # -- main loop ---------------------------------------------------------------------
    def run(self, task: str) -> RunRecord:
        record = RunRecord(task=task, model=self.model,
                           started_at=datetime.now().isoformat(timespec="seconds"))
        prompt = f"{self.system_prompt}\nUser Query: {task}\n"

        # Memoise successful calls: identical repeated Actions are the single most common
        # failure mode of small models and would otherwise burn the whole iteration budget.
        cache: Dict[str, str] = {}
        repair_counts: Dict[str, int] = {}
        # Steps that produced no new information: a cached repeat, a protocol violation,
        # or a call that failed again. A run of these means the model is stuck.
        unproductive = 0

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

            # 1. Terminal state -- but only if the numbers in it are real.
            if parsed["final_answer"]:
                answer = parsed["final_answer"]
                audit = audit_grounding(answer, self._evidence(record))
                record.grounding = audit

                if (is_fabricated(audit)
                        and record.grounding_rejections < self.max_grounding_rejections):
                    # The model wrote a results table for experiments it never ran.
                    # Refuse it and send it back to the loop rather than logging fiction.
                    record.grounding_rejections += 1
                    step.event = "grounding_rejected"
                    step.final_answer = answer
                    step.observation = json.dumps(audit)
                    record.steps.append(step)
                    demand = _REGROUND_DEMAND.format(
                        n=audit["numbers_ungrounded"],
                        values=", ".join(audit["ungrounded_values"]),
                        done=len(cache),
                    )
                    prompt += demand
                    self._say(demand.strip())
                    unproductive = 0  # give it a genuine chance to run the experiments
                    continue

                step.event = "final_answer"
                step.final_answer = answer
                record.steps.append(step)
                record.final_answer = answer
                record.completed = True
                if is_fabricated(audit):
                    self._say(f"\n>>> Task completed, but the answer is NOT grounded: "
                              f"{audit['numbers_ungrounded']}/{audit['numbers_claimed']} "
                              f"numbers appear in no Observation.")
                else:
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
                unproductive += 1
                if unproductive >= self.max_unproductive_steps:
                    break
                continue

            tool_name = parsed["action"]
            kwargs = parsed["action_input"]
            step.action, step.action_input = tool_name, kwargs

            # 3. Loop-breaker: serve an identical earlier call from cache.
            signature = f"{tool_name}::{json.dumps(kwargs, sort_keys=True, default=str)}"
            if signature in cache:
                step.event = "cached"
                step.observation = cache[signature]
                # Telling the model what it has NOT done yet is far more useful than
                # telling it to stop doing what it just did.
                used = {sig.split("::")[0] for sig in cache}
                unused = [t for t in TOOL_SPECS if t not in used]
                observation = (
                    f"\nObservation: {cache[signature]}\n"
                    "[NOTE] This exact call was already executed earlier in this run and "
                    "produced the result above. Calling it again cannot tell you anything "
                    "new.\n"
                    + (f"Tools you have NOT yet used: {unused}. If the task still needs "
                       "one of them, call it now.\n" if unused else "")
                    + "Otherwise give your Final Answer using the Observations above.\n"
                )
                prompt += observation
                record.steps.append(step)
                self._say(observation.strip())
                unproductive += 1
                if unproductive >= self.max_unproductive_steps:
                    break
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
                unproductive += 1
            else:
                cache[signature] = result
                unproductive = 0  # genuine new information: the model is making progress

            observation = f"\nObservation: {result}\n"
            if repair_hint:
                observation += repair_hint + "\n"
                self._say(repair_hint)
            prompt += observation
            record.steps.append(step)
            if unproductive >= self.max_unproductive_steps:
                break

        # The loop ended without the model volunteering a Final Answer -- either it stalled
        # or it ran out of iterations. Rather than abandoning a run whose Observations are
        # perfectly good, compel the conclusion it would not write on its own.
        if not record.completed:
            reason = ("stalled after "
                      f"{unproductive} unproductive steps" if unproductive >= self.max_unproductive_steps
                      else f"exhausted its {self.max_iterations}-iteration budget")
            self._say(f"\n>>> Agent {reason}. Forcing a Final Answer from the "
                      "observations already collected.")

            step = StepRecord(step=len(record.steps) + 1, event="forced_final")
            call_start = time.perf_counter()
            answer, meta = self._force_final_answer(prompt, self._evidence(record))
            step.llm_seconds = round(time.perf_counter() - call_start, 3)
            step.eval_count = int(meta.get("eval_count", 0))
            step.tokens_per_second = float(meta.get("tokens_per_second", 0.0))
            step.raw_output = answer
            step.final_answer = answer
            record.steps.append(step)

            evidence = self._evidence(record)
            audit = audit_grounding(answer, evidence) if answer else None

            # A forced answer gets no second lap round the loop, so it is verified here.
            # One retry with the constraint restated; a stalled run that never trained a
            # model will otherwise still report two confident model accuracies.
            if answer and is_fabricated(audit):
                self._say(">>> Forced answer failed the grounding audit "
                          f"({audit['ungrounded_values']}). Retrying once.")
                record.grounding_rejections += 1
                answer, meta = self._force_final_answer(prompt, evidence)
                step.raw_output = answer
                step.final_answer = answer
                audit = audit_grounding(answer, evidence) if answer else None

            record.grounding = audit
            if answer and not is_fabricated(audit):
                record.final_answer = answer
                record.completed = True
                record.forced_final = True
                self._say(f"\nFinal Answer: {answer}")
                self._say("\n>>> Task completed via forced finalisation.")
            elif answer:
                # Refuse to present invented results as the run's answer. The text is kept
                # in the record so the failure is auditable, but the run is not "completed".
                record.rejected_answer = answer
                record.completed = False
                step.event = "grounding_rejected"
                self._say(f"\n>>> REJECTED (ungrounded): {answer}")
                self._say(">>> Run did not produce a trustworthy answer: "
                          f"{audit['numbers_ungrounded']}/{audit['numbers_claimed']} "
                          "reported numbers appear in no Observation.")
            else:
                self._say("\n>>> Forced finalisation produced no answer.")

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
        f"Completed : {record.completed}"
        f"{' (forced finalisation)' if record.forced_final else ''}   "
        f"Steps: {len(record.steps)}   "
        f"Repairs: {record.repairs}   Parse errors: {record.parse_errors}   "
        f"Grounding rejections: {record.grounding_rejections}",
        f"Grounding : {json.dumps(record.grounding) if record.grounding else 'n/a'}",
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
