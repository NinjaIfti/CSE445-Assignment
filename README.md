# Autonomous Local LLM Machine Learning Agent

**CSE445: Machine Learning — Assignment #3**
North South University, Department of Electrical and Computer Engineering
Instructor: Dr. Mohammad Abdul Qayum (MAQm) · Section 6

A fully local, privacy-preserving **ReAct agent** that runs a quantized open-source LLM
through [Ollama](https://ollama.com) inside **WSL2** and autonomously orchestrates
Scikit-Learn and PyTorch machine learning pipelines. No paid APIs, no data leaves the
machine, and no agent framework — the Reason+Act loop is implemented from first
principles so every stage is inspectable.

---

## 1. What it does

Give it a sentence. It decides which experiments to run, runs them, reads the numbers,
recovers from its own mistakes, and reports a grounded conclusion.

```
$ python react_agent.py "Analyze the breast_cancer dataset, train a Random Forest and a
  PyTorch MLP on it, compare their accuracies, and recommend the best model."

--- Step 1 ---
Thought: I should first understand the structure of the dataset.
Action: load_dataset_summary
Action Input: {"dataset_name": "breast_cancer"}
Observation: {"status": "ok", "n_samples": 569, "n_features": 30, "n_classes": 2, ...}

--- Step 2 ---
Thought: Now I will train the Random Forest baseline.
Action: train_sklearn_model
Action Input: {"dataset_name": "breast_cancer", "model_type": "random_forest"}
Observation: {"status": "ok", "test_accuracy": 0.9561, "cv_mean_accuracy": 0.9543, ...}
...
```

---

## 2. Architecture

```
                    WINDOWS 11 HOST
  ┌──────────────────────────────────────────────────────────────┐
  │  NVIDIA driver  ──────────────────────────┐                  │
  └───────────────────────────────────────────┼──────────────────┘
                                              │ GPU passthrough
                    WSL2  (Ubuntu 22.04)      │ /usr/lib/wsl/lib
  ┌───────────────────────────────────────────┼──────────────────┐
  │                                           ▼                  │
  │   ┌────────────────────┐  HTTP POST  ┌──────────────────┐    │
  │   │  react_agent.py    │ ──────────► │  Ollama daemon   │    │
  │   │  ReAct controller  │             │  127.0.0.1:11434 │    │
  │   │                    │ ◄────────── │  llama3.2:3b     │    │
  │   └─────────┬──────────┘  completion │  (4-bit GGUF)    │    │
  │             │                        └──────────────────┘    │
  │             │ call_tool(name, kwargs)                        │
  │             ▼                                                │
  │   ┌────────────────────────────────────────────────────┐     │
  │   │            ml_tools.py — TOOL REGISTRY             │     │
  │   │  validation ▸ dispatch ▸ JSON envelope             │     │
  │   ├────────────────────────────────────────────────────┤     │
  │   │ load_dataset_summary   train_sklearn_model         │     │
  │   │ train_pytorch_mlp      tune_hyperparameters        │     │
  │   │ feature_selection      train_deep_classifier       │     │
  │   └──────────┬───────────────────────┬─────────────────┘     │
  │              ▼                       ▼                       │
  │      ┌──────────────┐        ┌──────────────┐                │
  │      │ Scikit-Learn │        │   PyTorch    │                │
  │      └──────────────┘        └──────────────┘                │
  └──────────────────────────────────────────────────────────────┘
```

### The controller loop

```
   user task
       │
       ▼
  ┌─────────────────┐
  │ build prompt    │◄────────────────────────────────────┐
  │ (system + tools │                                     │
  │  + scratchpad)  │                                     │
  └────────┬────────┘                                     │
           ▼                                              │
  ┌─────────────────┐    "Final Answer:"   ┌───────────┐   │
  │ query local LLM │─────────────────────►│   DONE    │   │
  │ stop=Observation│                      └───────────┘   │
  └────────┬────────┘                                      │
           ▼                                               │
  ┌─────────────────┐   unparseable   ┌──────────────────┐ │
  │ parse the turn  │────────────────►│ reformat nudge   │─┤
  └────────┬────────┘                 └──────────────────┘ │
           ▼ Action + Action Input                         │
  ┌─────────────────┐   seen before   ┌──────────────────┐ │
  │ loop-breaker    │────────────────►│ cached result    │─┤
  └────────┬────────┘                 └──────────────────┘ │
           ▼                                               │
  ┌─────────────────┐  status=error   ┌──────────────────┐ │
  │ execute tool    │────────────────►│ SELF-CORRECTION  │─┤
  │                 │                 │ hint + retry_with│ │
  └────────┬────────┘                 └──────────────────┘ │
           ▼ status=ok                                     │
      Observation ───────────────────────────────────────► ┘
```

| Component | Technology | Role |
|---|---|---|
| OS / subsystem | WSL2, Ubuntu 22.04 | Native Linux kernel, CUDA passthrough |
| Inference engine | Ollama, REST on `:11434` | Serves the 4-bit quantized model |
| Agent logic | Python 3.10+, custom ReAct controller | Prompting, parsing, dispatch, self-repair |
| ML frameworks | Scikit-Learn, PyTorch, Pandas, NumPy | Preprocessing, fitting, evaluation |

---

## 3. Quickstart

### Step 1 — Windows side (Administrator PowerShell, once)

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

Installs WSL2 + Ubuntu 22.04. **Reboot** when it asks, then launch Ubuntu once to create
your Linux user.

### Step 2 — Inside WSL

```bash
cd /mnt/k/'CSE445 Assignment'      # adjust to where you cloned it
bash setup_wsl.sh
```

This installs system packages, installs Ollama, starts the daemon, pulls `llama3.2:3b`,
creates a virtualenv, picks the correct PyTorch wheel (CUDA if a GPU is visible, else
CPU), installs dependencies and runs the test suite. It is idempotent — re-run it freely.

### Step 3 — Verify

```bash
source venv/bin/activate
python react_agent.py --health
```

```json
{ "reachable": true, "models": ["llama3.2:3b"], "requested_model_present": true }
```

---

## 4. Usage

```bash
# Ask the agent anything
python react_agent.py "Which model generalises best on wine, and by how much?"

# Generate the four required execution traces -> logs/*.log + logs/TRACES.md
python run_traces.py

# Deterministic reference benchmark (no LLM) -> docs/BENCHMARK.md
python benchmark_runner.py --mode direct

# Let the agent run that same benchmark itself -> docs/BENCHMARK_AGENT.md
python benchmark_runner.py --mode agent

# Profile local inference latency in WSL2 -> docs/LATENCY.md
python benchmark_runner.py --mode latency --repeats 5

# Exercise every tool without an LLM
python ml_tools.py
```

Useful flags: `--model mistral:7b`, `--max-iterations 20`, `--temperature 0.1`,
`--label my_run`, `--quiet`, `--no-log`.

Environment overrides: `OLLAMA_URL`, `OLLAMA_MODEL`.

---

## 5. Tool catalogue

The registry in `ml_tools.py` is the single source of truth — the system prompt is
generated from it, so the tool list the model sees can never drift from the code.

| # | Tool | Task | Purpose |
|---|---|---|---|
| 1 | `load_dataset_summary` | 1 | Shape, classes, balance, missing values, feature scale |
| 2 | `train_sklearn_model` | 1 | Decision tree / logistic regression / random forest + 5-fold CV |
| 3 | `train_pytorch_mlp` | 1 | One-hidden-layer MLP, full-batch Adam |
| 4 | `tune_hyperparameters` | 2 | GridSearchCV / RandomizedSearchCV over kernel SVM and decision trees |
| 5 | `feature_selection` | 2 | PCA and Sequential Feature Selection vs a full-feature baseline |
| 6 | `train_deep_classifier` | 2 | Deep MLP with Dropout, BatchNorm and cosine/step/plateau LR schedulers |

Datasets: `iris`, `wine`, `breast_cancer`.

Every tool returns a JSON **string**. Success is `{"status": "ok", ...}`; a recoverable
failure is `{"status": "error", "error_type": ..., "hint": ..., "retry_with": {...}}`.
Tools do not raise — that contract is what the self-correction loop is built on.

---

## 6. Self-correction

A 3B-parameter model gets things wrong constantly. Each failure mode is caught, typed,
and turned into a corrective instruction rather than a crash.

| `error_type` | Typical cause | What the agent is told |
|---|---|---|
| `unknown_tool` | Invented a tool name | Choose a name verbatim from the tool list |
| `unknown_dataset` | Asked for `titanic` | Retry with one of the three valid names |
| `unknown_model` | Asked `train_sklearn_model` for an SVC | Use `tune_hyperparameters` instead |
| `unexpected_argument` | Passed `n_estimators` to the wrong tool | Re-issue with the valid parameter names |
| `missing_argument` | Omitted `model_type` | Re-issue with every required parameter |
| `invalid_parameter` | `lr=50`, wrong type | Correct just that argument |
| `shape_mismatch` | 25 PCA components from 4 features | Reduce to fit the available features |
| `nan_loss` | Learning rate too high | Retry with a 10× smaller `lr` |
| `search_failed` | Invalid scoring string | Retry with default scoring |

Three further guards keep a small model from stalling:

- **Repair budget** — after 3 failures of the same call signature the agent is told to
  stop retrying and either change approach or answer with what it has.
- **Loop-breaker** — an identical successful call is served from cache with an explicit
  "you already ran this" note, instead of burning an iteration.
- **Reformat nudge** — output with no `Action:` and no `Final Answer:` triggers a
  minimal format reminder, and no tool is executed.

---

## 7. Testing

56 offline tests. They need **no Ollama and no network**: a `ScriptedLLM` replays fixed
model turns, so the parser, dispatcher, self-correction engine and loop-breaker are all
verified deterministically.

```bash
python -m pytest tests -q
```

```
56 passed
```

`tests/test_ml_tools.py` covers all six tools, determinism, and every error path.
`tests/test_react_agent.py` covers parsing (markdown fences, single quotes, function-call
syntax, bold headers), multi-tool chaining, self-correction, the repair budget, caching,
and log writing.

---

## 8. Repository map

```
├── ml_tools.py           # 6 ML tools + registry + validating dispatcher
├── react_agent.py        # ReAct controller, Ollama client, self-correction, logging
├── benchmark_runner.py   # direct / agent / latency benchmark modes
├── run_traces.py         # generates the 4 required execution traces
├── requirements.txt
├── setup_windows.ps1     # WSL2 install (Administrator, once)
├── setup_wsl.sh          # full in-WSL bootstrap (idempotent)
├── tests/
│   ├── test_ml_tools.py
│   └── test_react_agent.py
├── docs/
│   ├── REPORT.md         # technical report
│   ├── BENCHMARK.md      # generated reference benchmark
│   ├── BENCHMARK_AGENT.md# generated agent-authored benchmark
│   └── LATENCY.md        # generated WSL2 inference latency
└── logs/
    ├── TRACES.md         # generated index of all traces
    └── *.log / *.json    # generated per-run transcripts
```

---

## 9. Deliverables map

| Rubric item | Marks | Where |
|---|---|---|
| WSL & local LLM setup | 15 | `setup_windows.ps1`, `setup_wsl.sh`, `react_agent.py --health`, [REPORT §3](docs/REPORT.md) |
| ReAct loop & action parsing | 25 | `react_agent.py`, `tests/test_react_agent.py`, `logs/TRACES.md` |
| PyTorch & Scikit-Learn tools | 30 | `ml_tools.py`, `tests/test_ml_tools.py` |
| Statistical analysis & reasoning | 20 | `benchmark_runner.py`, `docs/BENCHMARK.md`, [REPORT §7](docs/REPORT.md) |
| Report & engineering quality | 10 | `docs/REPORT.md`, this README, the test suite |

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama at ...` | `ollama serve &` inside WSL; check `curl localhost:11434` |
| `Model 'llama3.2:3b' not pulled` | `ollama pull llama3.2:3b` |
| `torch.cuda.is_available()` is `False` | Update the **Windows** NVIDIA driver, not a Linux one. Re-run `setup_wsl.sh`; for RTX 50-series you need the `cu128` wheel |
| Agent loops without finishing | Raise `--max-iterations`, or lower `--temperature`; a larger model (`mistral:7b`) follows the protocol more reliably |
| Slow inference | Confirm GPU offload in `/tmp/ollama.log`; a 3B model on CPU runs at roughly 10–20 tok/s |
| `pip` cannot find a torch wheel | Your Python is too new. PyTorch wheels lag new releases — use Python 3.10–3.12 |
