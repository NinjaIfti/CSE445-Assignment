# Autonomous Local LLM Machine Learning Agent

**CSE445: Machine Learning — Assignment #3**
North South University, Department of Electrical and Computer Engineering
Instructor: Dr. Mohammad Abdul Qayum (MAQm) · Section 6
Primary reference: *Machine Learning with PyTorch and Scikit-Learn* — Raschka, Liu, Mirjalili

A fully local, privacy-preserving **ReAct agent** that runs a 4-bit quantized `llama3.2:3b`
through [Ollama](https://ollama.com) inside **WSL2** and autonomously orchestrates
Scikit-Learn and PyTorch pipelines. No paid APIs, no data leaves the machine, and no agent
framework — the Reason+Act loop is written from first principles so every stage is
inspectable and testable.

This README is also the technical report. Part I covers running the system; **Part II is
the report proper** (§1–§11).

---

## Contents

**Part I — Using it:** [Quickstart](#quickstart) · [Usage](#usage) · [Tool catalogue](#tool-catalogue) · [Testing](#testing) · [Repository map](#repository-map) · [Troubleshooting](#troubleshooting)

**Part II — Technical report:** [Summary of findings](#summary-of-findings) · [1 Objectives](#1-objectives-and-scope) · [2 Architecture](#2-system-architecture) · [3 Local LLM in WSL2](#3-local-llm-architecture-in-wsl2) · [4 Prompt engineering](#4-prompt-engineering) · [5 Parsing](#5-parsing-designing-for-a-model-that-will-not-comply) · [6 ML tools](#6-machine-learning-tool-implementation) · [7 Results & statistics](#7-experimental-results-and-statistical-analysis) · [8 Latency](#8-inference-latency-in-wsl2) · [9 Reliability](#9-reliability-self-correction-stall-breaking-and-grounding) · [10 Verification](#10-verification) · [11 Limitations](#11-limitations-and-future-work)

---

# Part I — Using it

## What it does

Give it a sentence. It decides which experiments to run, runs them, reads the numbers,
recovers from its own mistakes, and reports a conclusion grounded in real output.

```
$ python react_agent.py "Analyze the breast_cancer dataset, train a Random Forest and a
  PyTorch MLP on it, compare their accuracies, and recommend the best model."

--- Step 1 ---
Thought: To analyze the breast_cancer dataset, I need to load its summary statistics.
Action: load_dataset_summary(dataset_name="breast_cancer")
[latency: 0.751s | 65 tokens @ 148.11 tok/s]
Observation: {"status": "ok", "n_samples": 569, "n_features": 30, "n_classes": 2, ...}

--- Step 2 ---
Thought: Now I will train the Random Forest baseline.
Action: train_sklearn_model(dataset_name="breast_cancer", model_type="random_forest")
Observation: {"status": "ok", "test_accuracy": 0.9561, "cv_mean_accuracy": 0.9543, ...}
...
```

## Quickstart

### Step 1 — Windows side (Administrator PowerShell, once)

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot, then launch Ubuntu once to create your UNIX user.
`setup_windows.ps1` automates the same thing with extra GPU checks.

### Step 2 — inside WSL

```bash
cd /mnt/k/'CSE445 Assignment'
bash setup_wsl.sh
```

Installs system packages, Ollama, the daemon, `llama3.2:3b`, a virtualenv, the correct
PyTorch wheel (CUDA if a GPU is visible, else CPU), and runs the test suite. Idempotent.

### Step 3 — verify

```bash
source venv/bin/activate
python react_agent.py --health
```

```json
{ "reachable": true, "models": ["llama3.2:3b"], "requested_model_present": true }
```

## Usage

```bash
python react_agent.py "Which model generalises best on wine, and by how much?"

python run_traces.py --tag llama32-3b           # 6 traces -> logs/TRACES_llama32-3b.md
python run_traces.py --model mistral:7b --tag mistral7b   # same six on a 7B model
python benchmark_runner.py --mode direct        # reference benchmark -> docs/BENCHMARK.md
python benchmark_runner.py --mode agent         # agent runs it -> docs/BENCHMARK_AGENT.md
python benchmark_runner.py --mode latency -r 5  # WSL2 inference -> docs/LATENCY.md
python ml_tools.py                              # exercise every tool, no LLM needed
```

Flags: `--model mistral:7b`, `--max-iterations 20`, `--temperature 0.1`, `--label`,
`--quiet`, `--no-log`. Environment: `OLLAMA_URL`, `OLLAMA_MODEL`.

## Tool catalogue

The registry in `ml_tools.py` is the single source of truth — the system prompt is
generated from it, so the catalogue the model sees can never drift from the code.

| # | Tool | Task | Purpose |
|---|---|---|---|
| 1 | `load_dataset_summary` | 1 | Shape, classes, counts, proportions, missing values, feature scale |
| 2 | `train_sklearn_model` | 1 | Decision tree / logistic regression / random forest + 5-fold CV |
| 3 | `train_pytorch_mlp` | 1 | One-hidden-layer MLP, full-batch Adam |
| 4 | `tune_hyperparameters` | 2 | GridSearchCV / RandomizedSearchCV over kernel SVM and decision trees |
| 5 | `feature_selection` | 2 | PCA and Sequential Feature Selection vs a full-feature baseline |
| 6 | `train_deep_classifier` | 2 | Deep MLP with Dropout, BatchNorm, cosine/step/plateau LR schedulers, Adam or SGD |

Datasets: `iris`, `wine`, `breast_cancer`. Every tool returns a JSON **string** —
`{"status": "ok", ...}` or `{"status": "error", "error_type": ..., "hint": ...,
"retry_with": {...}}`. Tools do not raise; that contract is what self-correction is built on.

## Testing

**109 offline tests.** They need no Ollama and no network: a `ScriptedLLM` replays fixed
model turns, so the parser, dispatcher, self-correction engine, stall-breaker and
grounding audit are all verified deterministically.

```bash
python -m pytest tests -q
```
```
109 passed
```

| File | Tests | Covers |
|---|---|---|
| `tests/test_ml_tools.py` | 48 | All six tools, all three datasets, every scheduler, determinism, every error path |
| `tests/test_react_agent.py` | 52 | Parsing, multi-tool chaining, self-correction, stall-breaking, grounding audit, logging |
| `tests/test_statistics.py` | 9 | Corrected resampled t-test, paired CV, significance rendering |

## Repository map

```
├── ml_tools.py           # 6 ML tools + registry + validating dispatcher + sklearn wrapper
├── react_agent.py        # ReAct controller, Ollama client, self-correction, grounding audit
├── benchmark_runner.py   # direct / agent / latency modes + paired CV significance tests
├── run_traces.py         # generates the 6 execution traces
├── requirements.txt
├── setup_windows.ps1     # WSL2 install (Administrator, once)
├── setup_wsl.sh          # full in-WSL bootstrap (idempotent)
├── tests/                # 109 offline tests
├── docs/
│   ├── BENCHMARK.md      # generated: tool protocol + paired CV significance
│   ├── BENCHMARK_AGENT.md# generated: the agent's own benchmark table
│   └── LATENCY.md        # generated: WSL2 inference latency
└── logs/
    ├── TRACES_llama32-3b.md   # generated: six traces on the 3B model
    ├── TRACES_mistral7b.md    # generated: the same six on the 7B model
    └── *.log / *.json    # generated: per-run transcripts and structured records
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama at ...` | `ollama serve &` inside WSL; check `curl localhost:11434` |
| `Model 'llama3.2:3b' not pulled` | `ollama pull llama3.2:3b` |
| `torch.cuda.is_available()` is `False` | Update the **Windows** NVIDIA driver, not a Linux one. RTX 50-series needs the `cu128` wheel |
| Ollama installer: `requires zstd` | `sudo apt-get install zstd` (already in `setup_wsl.sh`) |
| Agent loops without finishing | Handled automatically (§9). Raise `--max-iterations` or use `mistral:7b` |
| `pip` cannot find a torch wheel | Python too new; PyTorch wheels lag. Use 3.10–3.12 |

---

# Part II — Technical Report

## Summary of findings

For a reader who wants the substance before the detail:

1. **A 3B model is not a reliable function caller.** Nearly every mechanism in §9 exists
   because `llama3.2:3b` failed in that specific way during a real run — stalling on a tool
   it had already called, emitting an empty answer, reading counts as percentages, and in
   one case inventing a complete results table. None of this was predictable from the
   offline test suite.
2. **The most dangerous failure produced no error at all** (§9.3). Asked to benchmark six
   model/dataset combinations, the agent ran one, then wrote a confident, well-formatted
   six-row table in which every figure was fabricated — including a clinical-screening
   recommendation. A grounding audit that checks each reported number against the actual
   observations now refuses such answers outright.
3. **Optimism bias is measurable and large** (§7.2). The tuned SVM's headline 0.9931 on
   wine falls to 0.9833 under nested cross-validation, and the random forest overtakes it.
   The 0.0098 gap is the cost of having selected hyperparameters on the evaluation folds.
4. **None of the algorithm differences is statistically significant** (§7.3). Under the
   Nadeau–Bengio corrected resampled t-test, all six pairwise comparisons across both
   datasets return p > 0.05. Reporting a winner from the fourth decimal place would be
   unsupportable.
5. **Model scale, not design, is the binding constraint** (§9.6). The identical prompt,
   tools and controller against `mistral:7b` completes 6 of 6 experiments with 17 of 17
   numbers grounded. The scaffolding lets a 3B model fail *safely*; it cannot make it
   capable.
6. **Nine defects in this project produced plausible output and were silently wrong**
   (§10) — a significance test that inverted its own verdict, a grounding audit that let
   fabrications validate themselves, and a `bool("false")` that quietly ran a different
   neural architecture than the one requested. None raised an exception.

## 1. Objectives and scope

This project implements an autonomous machine learning agent running entirely on local
hardware. A quantized open-source LLM served by Ollama inside WSL2 is the reasoning
engine; a hand-written ReAct controller turns its text output into calls against a
registry of six ML tools built on Scikit-Learn and PyTorch.

The engineering finding, stated plainly up front: **a 3-billion-parameter model is not a
reliable function caller, and essentially all of the value of this system lies in the
scaffolding that makes it one.** Every reliability mechanism in §9 exists because the
model failed in that specific way during real runs — not because it seemed prudent in
advance. Those failures, and the fixes, are the substance of this report.

Three deliverables follow the assignment tasks: a working environment and baseline ReAct
engine (Task 1); three additional ML tools (Task 2); self-correction plus an autonomous
model-comparison benchmark (Task 3).

## 2. System architecture

```
                    WINDOWS 11 HOST
  ┌──────────────────────────────────────────────────────────────┐
  │  NVIDIA driver (RTX 5060 Ti, sm_120) ─────┐                  │
  └───────────────────────────────────────────┼──────────────────┘
                                              │ GPU passthrough
                    WSL2  (Ubuntu 22.04.5)    │ /usr/lib/wsl/lib
  ┌───────────────────────────────────────────┼──────────────────┐
  │                                           ▼                  │
  │   ┌────────────────────┐  HTTP POST  ┌──────────────────┐    │
  │   │  react_agent.py    │ ──────────► │  Ollama daemon   │    │
  │   │  ReAct controller  │  /api/      │  127.0.0.1:11434 │    │
  │   │                    │  generate   │  llama3.2:3b     │    │
  │   │  • prompt builder  │ ◄────────── │  4-bit GGUF, 2GB │    │
  │   │  • output parser   │  completion │  llama.cpp core  │    │
  │   │  • self-correction │             └──────────────────┘    │
  │   │  • stall breaker   │                                     │
  │   │  • grounding audit │                                     │
  │   │  • run logger      │                                     │
  │   └─────────┬──────────┘                                     │
  │             │ call_tool(name, kwargs)                        │
  │             ▼                                                │
  │   ┌────────────────────────────────────────────────────┐     │
  │   │            ml_tools.py — TOOL REGISTRY             │     │
  │   ├────────────────────────────────────────────────────┤     │
  │   │  ① alias normalisation   dataset→dataset_name      │     │
  │   │  ② signature validation  reject unknown kwargs     │     │
  │   │  ③ required-arg check    reject incomplete calls   │     │
  │   │  ④ dispatch + exception → typed error envelope     │     │
  │   ├────────────────────────────────────────────────────┤     │
  │   │ load_dataset_summary   train_sklearn_model         │     │
  │   │ train_pytorch_mlp      tune_hyperparameters        │     │
  │   │ feature_selection      train_deep_classifier       │     │
  │   └──────────┬───────────────────────┬─────────────────┘     │
  │              ▼                       ▼                       │
  │      ┌──────────────┐        ┌──────────────┐                │
  │      │ Scikit-Learn │        │   PyTorch    │                │
  │      │ pipelines,   │        │ nn.Module,   │                │
  │      │ CV, search   │        │ Adam, sched. │                │
  │      └──────────────┘        └──────────────┘                │
  └──────────────────────────────────────────────────────────────┘
```

### 2.1 The controller loop

```
   user task
       │
       ▼
  ┌─────────────────┐
  │ build prompt    │◄──────────────────────────────────────────┐
  │ system + tool   │                                           │
  │ catalogue +     │                                           │
  │ scratchpad      │                                           │
  └────────┬────────┘                                           │
           ▼                                                    │
  ┌─────────────────┐                                           │
  │ query local LLM │                                           │
  │ stop=Observation│                                           │
  └────────┬────────┘                                           │
           ▼                                                    │
  ┌─────────────────┐   unparseable   ┌──────────────────┐      │
  │ parse the turn  │────────────────►│ reformat nudge   │──────┤
  └────────┬────────┘                 │ (no tool runs)   │      │
           │                          └──────────────────┘      │
           ├──── "Final Answer:" ───►┌──────────────────┐       │
           │                         │ GROUNDING AUDIT  │       │
           │                         │ every number in  │       │
           │                         │ an Observation?  │       │
           │                         └───┬──────────┬───┘       │
           │                        pass │          │ fabricated│
           │                             ▼          └───────────┤
           │                        ┌─────────┐                 │
           │                        │  DONE   │                 │
           │                        └─────────┘                 │
           ▼ Action + Action Input                              │
  ┌─────────────────┐   seen before   ┌──────────────────┐      │
  │ loop-breaker    │────────────────►│ cached result +  │──────┤
  │ (call cache)    │                 │ "do not repeat"  │      │
  └────────┬────────┘                 └──────────────────┘      │
           ▼                                                    │
  ┌─────────────────┐  status=error   ┌──────────────────┐      │
  │ execute tool    │────────────────►│ SELF-CORRECTION  │──────┤
  │ via call_tool   │                 │ strategy + hint  │      │
  │                 │                 │ + retry_with     │      │
  └────────┬────────┘                 └──────────────────┘      │
           ▼ status=ok                                          │
      Observation ──────────────────────────────────────────────┘
           │
           │  3 consecutive unproductive steps, or budget exhausted
           ▼
  ┌──────────────────────────────────────┐
  │ FORCED FINALISATION                  │
  │ pre-fill "Final Answer:" + enumerate │
  │ the only numbers it may quote        │
  └──────────────────────────────────────┘
```

| Component | Technology | Role |
|---|---|---|
| OS / subsystem | WSL2, Ubuntu 22.04.5, kernel 6.18 | Native Linux kernel, CUDA passthrough |
| Inference engine | Ollama, REST on `:11434` | Serves the 4-bit quantized model |
| Agent logic | Python 3.10.12, custom ReAct controller | Prompting, parsing, dispatch, self-repair |
| ML frameworks | Scikit-Learn 1.7, PyTorch 2.11+cu128 | Preprocessing, fitting, evaluation |

## 3. Local LLM architecture in WSL2

### 3.1 Why WSL2

WSL2 runs a real Linux kernel in a lightweight VM rather than emulating syscalls. Three
consequences matter here: Ollama ships as a Linux-first binary; CUDA reaches the guest
through the **Windows** driver mounted at `/usr/lib/wsl/lib` — installing a Linux NVIDIA
driver inside Ubuntu breaks passthrough, a common and costly mistake; and the `llama.cpp`
toolchain compiles without Windows-specific patches.

Verified on this machine: `nvidia-smi` inside Ubuntu reports the RTX 5060 Ti, and
`torch.cuda.is_available()` is `True` with compute capability `(12, 0)` — Blackwell, which
requires the `cu128` wheel. `setup_wsl.sh` detects the GPU and selects that index
automatically, falling back to the CPU build when no GPU is visible.

### 3.2 Quantization

`llama3.2:3b` is distributed as a **4-bit GGUF** file of 2.0 GB, against roughly 6.4 GB for
the same weights in FP16. Quantization maps each block of weights to a low-bit integer
with a shared scale:

$$w \approx s \cdot q, \qquad q \in \{0, \dots, 15\}, \qquad s = \frac{\max|w_{\text{block}}|}{15}$$

The `Q4_K_M` scheme keeps a few sensitive tensors (attention output, feed-forward
down-projections) at higher precision, which is why it degrades noticeably less than
uniform 4-bit. Practically: the model fits in GPU memory alongside the PyTorch training
jobs the agent launches, and inference stays interactive (§8).

### 3.3 Inference transport

The controller calls `POST /api/generate` with `stream: false`. Two options do most of the
work:

- `temperature: 0.1` — the task is protocol compliance, not creativity.
- `stop: ["Observation:", "\nObservation"]` — **the single most important setting.**
  Without it the model writes its own fabricated `Observation:` line and then reasons over
  invented numbers. The stop sequence forces it to yield control after every action.

That second setting also caused a bug worth recording. The forced-finalisation path (§9.2)
pre-fills `Final Answer:` and lets the model continue — but the model's first instinct
there is to write `Observation`, which tripped the stop immediately and returned an **empty
string**. The forced call now lifts the stop sequence explicitly. A safety mechanism
inherited a default that silently defeated it.

The response body also carries `eval_count`, `eval_duration`, `prompt_eval_count` and
`total_duration`, recorded per step to produce §8.

## 4. Prompt engineering

The system prompt is **generated from the tool registry** by `describe_tools()`, so the
catalogue can never drift from the code:

```
4. tune_hyperparameters(dataset_name, model_type, search_type="grid", cv=5, n_iter=20, scoring="accuracy")
   Run GridSearchCV or RandomizedSearchCV over an SVC (kernel SVM) or a decision_tree, ...
   Example Action Input: {"dataset_name": "wine", "model_type": "svc", "search_type": "grid"}
```

Techniques that measurably changed behaviour, in order of impact:

1. **A worked example per tool.** Small models pattern-match far better than they follow
   prose. This removed most argument-shape errors.
2. **An explicit anti-hallucination clause** — *"every number you report must come from an
   Observation."* Necessary but, as §9.3 shows, **not sufficient**.
3. **Explicit yield instruction** — *"Then STOP and wait."* Reinforces the stop sequence
   semantically.
4. **Error-handling rules stated in advance**, so the repair instructions the model later
   receives are familiar rather than novel.
5. **Enumerating permitted values.** In forced finalisation the prompt lists every number
   observed during the run and states that no other may appear. This converts open-ended
   generation into selection from a fixed set, and is dramatically more effective than any
   instruction not to invent numbers.

Defaults render as JSON (`true`, not `True`) so the model never sees Python syntax to imitate.

### 4.1 Naming is prompt engineering

The summary tool originally returned `class_balance: {"0": 59, "1": 71, "2": 48}`. The
model read those counts as **percentages** and reported "59% class 0, 71% class 1" — which
sums to 178%. The field is now `samples_per_class`, with a separate `class_proportions`
carrying the actual fractions. The model has reported class balance correctly since.

No prompt instruction fixed this; renaming the field did. Field names are read by the model
as documentation, and an ambiguous name is a prompting bug.

## 5. Parsing: designing for a model that will not comply

`parse_llm_output()` handles, in order of preference:

| Model output | Handling |
|---|---|
| Clean `Thought:` / `Action:` / `Action Input:` | Direct regex extraction |
| ` ```json {...} ``` ` fences | Fence stripped before parsing |
| `**Action:**` markdown bold | Optional `\**` in the header patterns |
| `{'dataset_name': 'iris'}` single quotes | `json.loads` fails → `ast.literal_eval` |
| `{"a": 1} and then I will...` trailing prose | Brace-balanced, string-aware scan |
| `train_sklearn_model(dataset_name="iris")` | Function-call regex fallback |
| `dataset_name="iris", epochs=50` no braces | `key=value` pair extraction |
| Conversational filler, no protocol | Reformat nudge; **no tool executed** |

The brace scanner is deliberately string-aware: a naïve `text[find("{"):rfind("}")+1]`
breaks on any argument value containing a brace. Each row is covered by a test.

### 5.1 Two parsing bugs that produced no error

Both were found by reading a real trace, not by the test suite, and both share a shape:
the call **succeeded** against arguments the model had not asked for.

The `nan_recovery` scenario asks for `hidden_dims=[64, 32]` and `batch_norm=false`. The
trace showed the tool receiving `hidden_dims: "[64"`. The `key=value` fallback matched
values with `[^,\n]+`, which stops at the first comma — including the comma *inside* the
list. The value scanner now tracks bracket depth and string state.

Worse, the same call recorded `batch_norm: true` despite the model asking for `false`.
Function-call syntax delivers the bare word `false` as the **string** `"false"`, and
`bool("false")` is `True`. The requested architecture was silently inverted, which is
precisely why training never diverged and the scenario never exercised the path it exists
to demonstrate. Booleans now go through `_coerce_bool`, which accepts the usual textual
forms and rejects anything else rather than guessing.

Neither bug raised an exception, failed a test, or appeared anywhere in the output. The
only symptom was a scenario that quietly did the wrong experiment.

### 5.2 When prose looks like protocol

A third trace bug is the sharpest of the set. The model produced a perfectly well-formed
turn:

```
Thought: ... I will retry the same Action with a learning rate ten times smaller.

Action: train_deep_classifier
Action Input: {"dataset_name": "breast_cancer", ...}
```

and the controller dispatched a tool called **`with`**. The header pattern was
`Action\s*:?\s*(\w+)` — with the colon *optional*, the phrase "the same Action with" inside
the Thought matched first, because `re.search` scans left to right and the prose comes
before the header.

The fix is to require the colon and anchor the header to the start of a line, which is what
actually distinguishes a protocol header from a sentence that happens to contain the word.
The same applied to `Action Input`. The hallucination firewall caught the consequence —
`unknown_tool` rather than a crash — but the agent still lost a step to a bug in the
component whose entire job is reading the model correctly.

The same trace showed a second problem: the generation was cut off mid-object at
`"lr": 0.` by the 512-token `num_predict` cap. That is a truncated generation, not
malformed JSON, and the parser now says so — telling the model to re-issue a shorter call
is actionable, while "invalid JSON" sends it hunting for a syntax error that is not there.
The cap is now 1024.

The function-call fallback earned its place immediately — in every real trace,
`llama3.2:3b` preferred `Action: load_dataset_summary(dataset_name="wine")` over the
documented two-line form, despite the prompt showing only the latter. Across the four
traces, **parse recoveries: 0** — not because the model followed the protocol, but because
the parser accommodated the protocol it actually used.

Beyond parsing, `call_tool()` is a **hallucination firewall** before any ML code runs. It
normalises argument synonyms (`dataset`→`dataset_name`, `model`→`model_type`,
`learning_rate`→`lr`), then rejects anything still unrecognised with the valid signature
attached. Normalisations are echoed back as `_normalised_arguments`, so a trace never hides
a silent rewrite.

## 6. Machine learning tool implementation

### 6.1 Task 1 — baseline

`load_dataset_summary`, `train_sklearn_model` (decision tree, logistic regression, random
forest) and `train_pytorch_mlp`. Two details beyond the assignment skeleton: logistic
regression is wrapped in a `Pipeline` with `StandardScaler`, because `breast_cancer`
features span four orders of magnitude and an unscaled fit is a convergence problem rather
than a modelling result; and standardisation in the PyTorch tool uses **training statistics
only**, since computing the mean over the full array before splitting leaks test
information.

### 6.2 Task 2 — advanced tools

**`tune_hyperparameters`** — `GridSearchCV` or `RandomizedSearchCV` over an SVC or decision
tree. The estimator sits inside a `Pipeline` with a scaler, so the scaler is refit on every
CV fold; scaling once before cross-validation leaks fold statistics.

| Model | Grid | Candidates |
|---|---|---|
| SVC | `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {scale, auto, 0.01, 0.1, 1}`, `kernel ∈ {rbf, poly, linear}` | 60 |
| Decision tree | `max_depth ∈ {2,3,4,6,8,None}`, `min_samples_split ∈ {2,5,10}`, `min_samples_leaf ∈ {1,2,4}`, `criterion ∈ {gini, entropy}` | 108 |

It returns a top-3 leaderboard, not just the winner, so the agent can reason about how
sensitive the result is.

**`feature_selection`** — PCA and `SequentialFeatureSelector`, each scored against an
identical full-feature logistic-regression baseline so the accuracy cost of compression is
measured, not assumed. PCA maximises retained variance by projecting onto the top
eigenvectors of the covariance matrix; SFS greedily optimises cross-validated accuracy
directly. They answer different questions and both are reported.

**`train_deep_classifier`** — a configurable `[Linear → BatchNorm → ReLU → Dropout] × N →
Linear` stack with cosine, step, plateau or no LR schedule, optional weight decay, and a
choice of Adam or SGD. Cosine annealing follows

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\left(\tfrac{t}{T}\pi\right)\right)$$

It reports `generalisation_gap` = train − test accuracy, the quantity Dropout and BatchNorm
exist to control. Reporting it makes over-fitting visible to the agent rather than leaving
it to be inferred.

**Why the optimizer choice exists.** Task 3 requires the agent to recover from a NaN loss,
which means divergence has to be genuinely reachable. It is not with Adam: Adam rescales
each gradient by its running second moment, and even at `lr=9.9` — the top of the accepted
range — it converges. Plain SGD has no such protection and, without BatchNorm, diverges
reliably at that step size. Adding the optimizer is therefore not decoration: it is what
makes the `nan_loss` path testable end-to-end rather than only under a monkeypatch, and
optimiser comparison is textbook material in its own right.

The recovery hint needed care too. "Retry with a 10× smaller learning rate" sounds
reasonable and is **wrong here**: from `lr=9.9` it lands on 0.99, which limps to 0.63
accuracy, while 0.5 diverges again and 0.1 trains cleanly to 0.9561. Recovery is not
monotonic in the learning rate. The suggestion is therefore clamped to a value known to be
stable for the optimiser in use (0.1 for SGD, 0.01 for Adam), and a test asserts that
following the hint *actually converges* — a hint that does not fix the problem is worse
than no hint, because the agent will faithfully follow it into a second failure.

### 6.3 Reproducibility

Every stochastic tool re-seeds NumPy and PyTorch on entry; all splits use
`random_state=42` with stratification. Both PyTorch tools accept a `seed` controlling
initialisation and batch order **while holding the split fixed**, which isolates
optimisation variance from split variance.

`TorchMLPClassifier` wraps the network in the Scikit-Learn estimator API (`fit`/`predict`,
`clone`-safe). Comparing a neural network against classical estimators is only meaningful
if both are scored on identical folds; the wrapper is what makes §7.2 possible.

## 7. Experimental results and statistical analysis

### 7.1 Tool protocol — what the agent sees

From `python benchmark_runner.py --mode direct`. Scikit-Learn models use 5-fold stratified
CV; the network is retrained under 5 seeds on a fixed split.

| Dataset | Algorithm | Val accuracy (mean ± std) | Test accuracy |
|---|---|---|---|
| wine | Random Forest | 0.9775 ± 0.0213 | 1.0000 |
| wine | Kernel SVM (tuned) | **0.9931 ± 0.0138** | 0.9444 |
| wine | Deep MLP (Dropout+BN) | 0.9666 ± 0.0111 | 0.9722 |
| breast_cancer | Random Forest | 0.9543 ± 0.0150 | 0.9561 |
| breast_cancer | Kernel SVM (tuned) | **0.9758 ± 0.0108** | 0.9825 |
| breast_cancer | Deep MLP (Dropout+BN) | 0.9737 ± 0.0096 | 0.9825 |

Note the wine random forest scoring a *perfect* 1.0000 on the hold-out test set while
ranking second on cross-validation. With a 36-row test set, 1.0000 and 0.9722 differ by one
sample. This is precisely why the cross-validated mean, not the hold-out number, is the
comparison used below.

### 7.2 Paired cross-validation — the rigorous comparison (CO2)

The table above is not a fair comparison: each algorithm is scored under its own protocol,
and the SVM's hyperparameters were selected using the same folds it was then scored on.
So all three algorithms were re-run on the **same 10 stratified folds**, with the SVM search
nested *inside* each outer training fold (15 candidates, 3-fold inner CV).

| Dataset | Algorithm | Mean accuracy | Std | Per-fold range |
|---|---|---|---|---|
| wine | **Random Forest** | **0.9889** | 0.0222 | 0.9444 – 1.0000 |
| wine | Kernel SVM (tuned) | 0.9833 | 0.0255 | 0.9444 – 1.0000 |
| wine | Deep MLP (Dropout+BN) | 0.9830 | 0.0260 | 0.9412 – 1.0000 |
| breast_cancer | **Kernel SVM (tuned)** | **0.9772** | 0.0193 | 0.9474 – 1.0000 |
| breast_cancer | Deep MLP (Dropout+BN) | 0.9701 | 0.0176 | 0.9474 – 1.0000 |
| breast_cancer | Random Forest | 0.9526 | 0.0314 | 0.8772 – 0.9825 |

**The two protocols disagree, and the rigorous one is right.** On wine the tuned SVM looked
like the clear winner at 0.9931; under nested CV it falls to 0.9833 and the random forest
leads at 0.9889. That 0.0098 drop is **optimism bias** — the price of having selected
hyperparameters on the evaluation folds. This is the single most useful result in the
project, and it is invisible unless the comparison is done properly.

### 7.3 Are the differences real?

Differences are tested with the **Nadeau–Bengio corrected resampled t-test**, which
Raschka recommends for exactly this situation. The ordinary paired t-test assumes
independent folds; CV folds share training data, so the naive test understates variance and
over-claims significance. The correction inflates it:

$$t = \frac{\bar{d}}{\sqrt{\left(\frac{1}{k} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)\sigma_d^2}}, \qquad \frac{n_{\text{test}}}{n_{\text{train}}} = \frac{1}{k-1}$$

| Dataset | Comparison | Mean diff | t | p | Verdict (α=0.05) |
|---|---|---|---|---|---|
| wine | RF vs SVM | +0.0056 | 0.383 | 0.7103 | not significant |
| wine | RF vs Deep MLP | +0.0059 | 0.309 | 0.7644 | not significant |
| wine | SVM vs Deep MLP | +0.0003 | 0.019 | 0.9853 | not significant |
| breast_cancer | RF vs SVM | −0.0246 | −1.718 | 0.1199 | not significant |
| breast_cancer | RF vs Deep MLP | −0.0175 | −0.984 | 0.3507 | not significant |
| breast_cancer | SVM vs Deep MLP | +0.0070 | 0.744 | 0.4757 | not significant |

**None of the six comparisons is significant.** The honest conclusion is that all three
algorithms are statistically indistinguishable on both datasets at these sample sizes — a
far more defensible answer than picking a winner from the fourth decimal place. On a
representative pair, the corrected test gives p = 0.0036 where the uncorrected paired
t-test gives p = 0.0003: an order of magnitude more conservative, and the reason
"significant" differences in small-data ML papers so often fail to replicate.

### 7.4 Interpretation (CO1)

The result is unglamorous and worth stating: **on tabular problems of a few hundred rows, a
tuned kernel SVM and a random forest match a regularised deep network**, at a fraction of
the training cost. The network needs 3,267–4,322 parameters to fit datasets of 178–569
samples; its capacity is not the binding constraint — sample size is. The measured
generalisation gaps (0.033 wine, 0.022 breast_cancer) show Dropout and BatchNorm doing
their job, but controlling over-fitting cannot manufacture information the data does not
contain.

This mirrors the textbook's framing: model selection is bounded by the bias–variance
trade-off and the data available, not by architectural sophistication.

## 8. Inference latency in WSL2

From `python benchmark_runner.py --mode latency --repeats 5`, measuring wall-clock
alongside Ollama's own `eval_count` / `eval_duration` counters.

| Prompt type | Mean (s) | Std (s) | Min (s) | Max (s) | Tokens | Tok/s |
|---|---|---|---|---|---|---|
| short | 0.672 | 1.302 | 0.016 | 3.277 | 2.8 | 231.04 |
| medium | 0.744 | 0.044 | 0.686 | 0.797 | 120 | 167.24 |
| react_turn | 0.119 | 0.015 | 0.108 | 0.148 | 17.6 | 173.29 |

Hardware: RTX 5060 Ti (8 GB), 15 GB RAM allocated to WSL2.

The `short` row's std of 1.302 s exceeds its mean — an artefact worth reading correctly.
The first call includes a one-time model load of 3.277 s; the remaining four run in 0.016 s.
Quoting "0.672 s mean" for this row would be actively misleading. The steady-state figure
is **167–174 tok/s with GPU offload**, roughly 10× the 10–20 tok/s typical of CPU-only
inference for a 4-bit 3B model.

**Where the wall-clock time actually goes.** Tool execution dominates, not inference. In the
`full_pipeline` trace, 7 LLM calls consumed 4.7 s while tools consumed 59.5 s — the
60-candidate grid search is simply expensive. But inference cost grows with every step
because the scratchpad is re-sent in full, which is the direct argument for the loop-breaker
and repair budget in §9: each suppressed redundant call saves an entire forward pass over a
monotonically growing prompt.

## 9. Reliability: self-correction, stall-breaking and grounding

This section is the substance of Task 3. Each mechanism below was added in response to an
observed failure of `llama3.2:3b`, not anticipated in advance.

### 9.1 Self-correction from typed errors

Tools never raise for recoverable problems. They return a typed envelope:

```json
{
  "status": "error",
  "error_type": "shape_mismatch",
  "message": "n_components=25 is invalid for iris, which has only 4 features.",
  "hint": "n_components must satisfy 1 <= n_components <= 4.",
  "retry_with": {"dataset_name": "iris", "method": "pca", "n_components": 4}
}
```

`_build_repair_hint()` converts it into an instruction appended after the observation.
Nine error types are mapped: `unknown_tool`, `unknown_dataset`, `unknown_model`,
`unexpected_argument`, `missing_argument`, `invalid_parameter`, `shape_mismatch`,
`nan_loss` (retry at one tenth the learning rate) and `search_failed`.

The design choice worth defending is **`retry_with`**. Handing back a corrected argument
dictionary rather than only a description is what makes recovery reliable at 3B scale: the
model's job collapses from *diagnose and re-derive* to *copy*. In the `self_correction`
trace this works on the very next step — the agent requests 25 components from a 4-feature
dataset, reads the hint, and retries with 4.

It also shows the limit. Having succeeded, the model then reverts to requesting 25 again.
Recovery is reliable; *learning* from the recovery is not. Hence the bounds below.

### 9.2 Stall-breaking and forced finalisation

The most common real failure was not a crash. In the first trace run, the agent called
`load_dataset_summary`, received a perfect observation, and then called it **five more
times**, ignoring the "you already ran this" note each time, until the iteration budget
expired with no answer at all.

Three bounds now apply: a repair budget of three attempts per call signature; a cache that
intercepts repeated successful calls; and a **stall detector** that counts consecutive
unproductive steps (cached repeat, protocol violation, or repeated failure) and resets on
genuine progress.

After three unproductive steps, the controller stops asking and **pre-fills the answer**:
it appends `Thought: I have gathered all necessary experimental data.\nFinal Answer:` to the
prompt. Because `/api/generate` continues whatever text it is given, the model has no
syntactic room to emit another Action — it can only write the answer, and the observations
are still in the scratchpad to ground it. Every one of the four scenarios now reaches a
Final Answer; before this mechanism, two did not.

### 9.3 The grounding audit

The most dangerous failure produced no error at all. Asked to benchmark 3 algorithms across
2 datasets, the agent ran **five steps** — nowhere near the six experiments required — and
emitted a complete, well-formatted 6-row Markdown results table:

| Dataset | Algorithm | CV/Val Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| Wine | Random Forest | 0.94 | 0.93 |
| Wine | Kernel SVM | 0.95 | 0.94 |
| ... | ... | ... | ... |

Every number was invented. None appears in any observation from the run. The output looked
more authoritative than the genuine answers, and nothing about it signalled failure. The
prompt's explicit "never invent numerical results" instruction did not prevent it.

`audit_grounding()` therefore checks every decimal in a Final Answer against the numbers
that actually appeared in observations, at the precision the answer states, accepting both
rounding (0.96 from an observed 0.9561) and percentage rendering (97.42 from 0.9742). An
answer with three or more unaccounted numbers, and under half its figures grounded, is
rejected: the run continues with an explicit demand to execute the missing experiments.
Every trace records its grounding ratio, so an ungrounded answer is visible rather than
silently trusted.

Three subtleties, each found by running the thing rather than by inspection:

- The rejected answer's audit was initially stored as that step's observation. Since the
  audit necessarily quotes the invented values, they became "evidence" on the next pass —
  a fabricated table validating itself. Audit steps are now excluded from the evidence set.
- The audit flagged `97.42%` as invented when the observation held `0.9742`. That is
  faithful reporting, not fabrication, so proportion→percentage conversion is accepted.
  It still catches the genuine case: an answer claiming "92.68% of variance retained" when
  the observation reported `0.7268`.
- **A forced answer gets no second lap of the loop, so it must be verified in place.** A
  later `multi_tool` run stalled having called only `load_dataset_summary` — it trained
  nothing — and its forced answer still reported "Random Forest 97.4%, PyTorch MLP 96.8%"
  and recommended one for *clinical screening*. Two invented numbers fell under the
  original three-number threshold. The rule now also fires when **nothing** is grounded
  across two or more values, and a fabricated forced answer is regenerated once; if it
  fabricates again the run is marked **not completed**, with the text retained in
  `rejected_answer` for audit. Refusing to answer is the correct outcome — presenting
  invented clinical accuracies as a result is not.

A related fix went into the stall-breaker. The cached-repeat note used to say only "do not
repeat this call". It now enumerates the tools the run has **not** yet used, because
telling a small model what remains to be done is far more actionable than telling it to
stop doing what it just did.

### 9.4 Observed behaviour

Six scenarios were run against both models — twelve traces in total. **All twelve reach a
Final Answer, and all twelve pass the grounding audit with every reported number traced
back to an observation.**

| Scenario | Demonstrates | 3B steps / corrections | 7B steps / corrections |
|---|---|---|---|
| `single_tool` | Baseline ReAct cycle | 6 / 0 | 6 / 0 |
| `multi_tool` | Chaining three tools | 10 / 2 | 4 / 0 |
| `self_correction` | **Shape mismatch** recovery | 7 / 1 | 3 / 1 |
| `nan_recovery` | **NaN loss** recovery | 8 / 1 | 3 / 1 |
| `parameter_error` | **Parameter error** recovery | 6 / 0 | 6 / 4 |
| `full_pipeline` | Tuning + PCA + deep net | 8 / 0 | 4 / 0 |

The three failure modes Task 3 names are each demonstrated end-to-end. `nan_recovery` is
the cleanest: both models request `lr=9.9`, receive `nan_loss`, read the clamped
suggestion, retry at `lr=0.1`, and reach 0.9474 test accuracy.

The difference between the models is not success versus failure — it is *directness*. The
3B model needs 45 steps across the six scenarios where the 7B needs 26, and it wanders
after succeeding: in `nan_recovery` it recovers at step 2, then re-calls tools it has
already run four more times before the stall-breaker forces its conclusion. The 7B model
recovers and stops. Both produce grounded answers; one wastes two-thirds of its budget
doing so.

The fix in §9.2 is visible in the numbers. Before the "tools you have not yet used" hint,
`multi_tool` stalled after 4 steps having called a single tool. After it, the same scenario
runs 10 steps and 8 tool calls, actually training both models before answering.

Indexes: `logs/TRACES_llama32-3b.md` and `logs/TRACES_mistral7b.md`.

### 9.5 Where the 3B model still fails

The Task 3 benchmark prompt — three algorithms across two datasets, six experiments in one
instruction — is beyond `llama3.2:3b`. Across repeated runs it calls
`load_dataset_summary`, drifts, and never completes the experiment matrix. With the
grounding audit in place the failure is at least *honest*: its final answer is now

> "The test accuracy, CV accuracy, and standard deviation for the breast_cancer dataset are
> not available in the Observation report."

— zero numbers claimed, nothing invented. Before the audit, the same prompt produced a
confident, fully-populated six-row results table in which **every figure was fabricated**.

That is the correct engineering outcome and the wrong user outcome: the model can be made
to stop lying, but it cannot thereby be made competent.

### 9.6 The same prompt at 7B

The obvious question is whether this is a limit of the design or of the model. It is the
model. Running the identical prompt, identical tools and identical controller against
`mistral:7b` — changing only `--model` — the agent completes the task:

| | `llama3.2:3b` | `mistral:7b` |
|---|---|---|
| Distinct experiments run | 1 of 6 | **6 of 6** |
| Tool calls | 3, all `load_dataset_summary` | 6, correctly distributed |
| Reached Final Answer | only when forced | **volunteered** |
| Numbers claimed / ungrounded | 14 / 7, then 0 / 0 after refusal | **17 / 0** |
| Produced the required table | no | **yes, with both analysis paragraphs** |

Mistral called `train_sklearn_model`, `tune_hyperparameters` and `train_deep_classifier`
across both datasets in order, then wrote the comparison table from the observations it
had actually collected — including the correct observation that most of the differences
are not significant given the reported standard deviations. Its output is
`docs/BENCHMARK_AGENT.md`.

Two things are worth drawing out. First, **the scaffolding is what makes the comparison
meaningful**: without the grounding audit, the 3B run would have produced a table that
*looked* just as complete as Mistral's, and the difference between the two models would
have been invisible. Second, the audit passed the 7B answer untouched — it is a filter on
fabrication, not a tax on competence.

The reference numbers in §7 still come from `--mode direct`, which executes the same matrix
deterministically and adds the nested-CV statistics no model produces on its own.

## 10. Verification

**109 tests, no Ollama and no network required.**

```
$ python -m pytest tests -q
109 passed
```

A `ScriptedLLM` replays fixed model turns, so behaviours that would otherwise need a live
model are deterministic. `test_agent_self_corrects_a_shape_mismatch` asserts not merely that
the run finishes but that the corrective text and the exact `retry_with` payload reached the
model on the following turn;
`test_fabricated_final_answer_is_rejected_and_sent_back_to_the_loop` reproduces the §9.3
failure and asserts the recovery.

Defects caught by tests or by reading real traces, rather than by inspecting code:

| Defect | Symptom | Why it mattered |
|---|---|---|
| `latency_summary` filtered on `> 0` | Sub-millisecond calls vanished from the stats | Under-reported call counts |
| Corrected t-test's zero-variance branch | "Not significant" for two models differing by the same margin on *every* fold | Exactly inverted the verdict |
| Grounding audit stored as an observation | Invented values became evidence next pass | A fabricated table validated itself |
| Fabrication threshold used `< 0.5` | An answer at exactly 0.5 (7 of 14 invented) passed | Off-by-one let fiction through |
| `Observation:` stop left on for forced answers | Empty answer every time (§3.3) | A safety default defeated a safety mechanism |
| `[^,\n]+` value pattern | `hidden_dims=[64, 32]` became `"[64"` (§5.1) | Ran a different architecture, silently |
| `bool("false")` | `batch_norm=false` became `True` (§5.1) | Inverted the request; the NaN scenario never diverged |
| Optional colon in the `Action` header | Prose "the same Action with" dispatched a tool named `with` (§5.2) | Lost a step to the parser itself |
| `num_predict: 512` | Action Input truncated mid-object | A valid call lost to the token budget |

The pattern is worth naming: **every one of these produces plausible output and is wrong.**
None raised an exception. That is the class of bug offline reasoning does not surface, and
it is the argument for running the real model early rather than after the code "looks
finished".

**Environment:** Ubuntu 22.04.5 LTS on WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2),
Python 3.10.12, PyTorch 2.11.0+cu128 with CUDA available, NVIDIA RTX 5060 Ti (sm_120),
Ollama serving `llama3.2:3b` (4-bit GGUF, 2.0 GB). All ML results are seed-fixed and
reproduce on re-run; latency figures are hardware-dependent.

## 11. Limitations and future work

- **Model scale is the binding constraint**, and §9.6 measures it rather than asserting it:
  on the six-experiment benchmark `llama3.2:3b` completes 1 of 6 experiments and invents
  the rest, while `mistral:7b` completes 6 of 6 with every number grounded. The reliability
  scaffolding is what lets a 3B model fail safely; it does not make it capable.
- **The scratchpad grows without bound.** Each step re-sends the entire history. Beyond a
  dozen steps this dominates latency; summarising older observations, or using Ollama's
  `context` token array, would bound it.
- **Grounding is checked numerically, not semantically.** The audit confirms a number
  appeared in *some* observation, not that it was attached to the right model or dataset.
  An answer that swaps two correctly-observed accuracies would pass.
- **Three datasets, all small, all clean.** No missing values, no categorical encoding, no
  real class imbalance. The `missing_values` field is scaffolding for data that does not
  yet exercise it.
- **Constrained decoding is unused.** Ollama supports grammar- and JSON-schema-constrained
  generation, which would eliminate the §5 parsing failure modes outright rather than
  recovering from them.

---

## Deliverables map

| Rubric item | Marks | Where |
|---|---|---|
| WSL & local LLM setup | 15 | `setup_windows.ps1`, `setup_wsl.sh`, `react_agent.py --health`, [§3](#3-local-llm-architecture-in-wsl2) |
| ReAct loop & action parsing | 25 | `react_agent.py`, `tests/test_react_agent.py`, [§5](#5-parsing-designing-for-a-model-that-will-not-comply), `logs/TRACES_*.md` |
| PyTorch & Scikit-Learn tools | 30 | `ml_tools.py`, `tests/test_ml_tools.py`, [§6](#6-machine-learning-tool-implementation) |
| Statistical analysis & reasoning | 20 | `benchmark_runner.py`, `docs/BENCHMARK.md`, [§7](#7-experimental-results-and-statistical-analysis) |
| Report & engineering quality | 10 | This document, the 109-test suite |

### Task coverage against the brief

| Task | Requirement | Where |
|---|---|---|
| 1 (25) | WSL2 + Ollama + venv with full GPU/CPU PyTorch support | `setup_windows.ps1`, `setup_wsl.sh` (auto-selects `cu128`), §3.1 |
| 1 | Baseline ReAct loop, single-tool and multi-tool traces verified | `single_tool` and `multi_tool` in `logs/TRACES_*.md` |
| 2 (40) | Hyperparameter tuning: GridSearchCV / RandomizedSearchCV for SVC and Decision Trees | `tune_hyperparameters`, §6.2 |
| 2 | Feature selection & dimensionality reduction: PCA + sequential feature selection | `feature_selection`, §6.2 |
| 2 | Deep PyTorch classifier with Dropout, BatchNorm, custom LR schedulers | `train_deep_classifier`, §6.2 |
| 3 (35) | Self-healing on **shape mismatch** | `self_correction` trace |
| 3 | Self-healing on **parameter error** | `parameter_error` trace |
| 3 | Self-healing on **NaN loss** | `nan_recovery` trace (SGD divergence, §6.2) |
| 3 | Comprehensive prompt: 3 algorithms × 2 datasets, CV, Markdown summary table | `BENCHMARK_TASK` in `benchmark_runner.py` → `docs/BENCHMARK_AGENT.md`, §9.6 |

| Required deliverable | Status |
|---|---|
| GitHub repo with `ml_tools.py`, `react_agent.py`, `benchmark_runner.py`, `requirements.txt` | ✅ |
| Execution logs: ≥3 multi-step reasoning traces with tool execution | ✅ 4 traces in `logs/` |
| Report: local LLM architecture, prompt engineering, WSL2 latency | ✅ [§3](#3-local-llm-architecture-in-wsl2), [§4](#4-prompt-engineering), [§8](#8-inference-latency-in-wsl2) |
| Report: mathematical model comparison (CO1, CO2) | ✅ [§7](#7-experimental-results-and-statistical-analysis) |
| Report: architectural diagram of controller loop and tool registry | ✅ [§2](#2-system-architecture) |
