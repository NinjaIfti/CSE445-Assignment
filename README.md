# Building an Autonomous Local LLM Machine Learning Agent in Windows WSL

**CSE445: Machine Learning — Assignment #3**
North South University · Department of Electrical and Computer Engineering
Instructor: Dr. Mohammad Abdul Qayum (MAQm) · Section 6
Primary reference: *Machine Learning with PyTorch and Scikit-Learn* — Raschka, Liu, Mirjalili

A fully local, privacy-preserving **ReAct agent**: a 4-bit quantized LLM served by Ollama
inside WSL2 reasons about a natural-language request, then drives Scikit-Learn and PyTorch
tools to answer it. No paid APIs, no data leaves the machine, and no agent framework — the
Reason+Act loop is implemented from first principles.

---

## Submission contents

| Deliverable | Location |
|---|---|
| **Code** | [`ml_tools.py`](ml_tools.py), [`react_agent.py`](react_agent.py), [`benchmark_runner.py`](benchmark_runner.py), [`requirements.txt`](requirements.txt) |
| **Execution logs** (≥3 required, 12 provided) | [`logs/TRACES_llama32-3b.md`](logs/TRACES_llama32-3b.md), [`logs/TRACES_mistral7b.md`](logs/TRACES_mistral7b.md) + per-run `.log`/`.json` |
| **Technical report** | [Part II](#part-ii--technical-report) of this file |
| Generated results | [`docs/BENCHMARK.md`](docs/BENCHMARK.md), [`docs/BENCHMARK_AGENT.md`](docs/BENCHMARK_AGENT.md), [`docs/LATENCY.md`](docs/LATENCY.md) |
| Supporting detail | [`docs/APPENDIX.md`](docs/APPENDIX.md) |

### Task coverage

| Task | Requirement | Delivered |
|---|---|---|
| **1** (25) | WSL2 + Ollama + venv, full GPU/CPU PyTorch | `setup_windows.ps1`, `setup_wsl.sh`; verified on Ubuntu 22.04.5, torch 2.11.0+cu128, CUDA available |
| **1** | Baseline ReAct loop; single- and multi-tool traces | `single_tool`, `multi_tool` traces |
| **2** (40) | Hyperparameter tuning: Grid/RandomizedSearchCV for SVC and Decision Trees | `tune_hyperparameters` |
| **2** | Feature selection & dimensionality reduction: PCA + sequential selection | `feature_selection` |
| **2** | Deep PyTorch classifier: Dropout, BatchNorm, custom LR schedulers | `train_deep_classifier` |
| **3** (35) | Self-healing on **shape mismatch** | `self_correction` trace |
| **3** | Self-healing on **parameter error** | `parameter_error` trace |
| **3** | Self-healing on **NaN loss** | `nan_recovery` trace |
| **3** | 3 algorithms × 2 datasets, cross-validated, Markdown summary table | `docs/BENCHMARK_AGENT.md` |

---

## Setup

**Step 1 — Windows, Administrator PowerShell (once).** Reboot afterwards, then launch
Ubuntu once to create your UNIX user.

```powershell
wsl --install -d Ubuntu-22.04
```

**Step 2 — inside WSL.** Installs system packages, Ollama, the model, a virtualenv and the
correct PyTorch wheel (CUDA if a GPU is visible, else CPU), then runs the test suite.
Idempotent — safe to re-run.

```bash
cd /mnt/k/'CSE445 Assignment'
bash setup_wsl.sh
```

**Step 3 — verify.**

```bash
source venv/bin/activate
python react_agent.py --health     # {"reachable": true, "requested_model_present": true}
python -m pytest tests -q          # 109 passed
```

## Running it

```bash
python react_agent.py "Compare a random forest and a deep MLP on breast_cancer."

python run_traces.py --tag llama32-3b                     # 6 traces -> logs/
python run_traces.py --model mistral:7b --tag mistral7b   # the same six on a 7B model
python benchmark_runner.py --mode direct                  # reference benchmark + statistics
python benchmark_runner.py --mode agent                   # the agent runs the benchmark
python benchmark_runner.py --mode latency --repeats 5     # WSL2 inference latency
python ml_tools.py                                        # exercise every tool, no LLM
```

## Repository structure

```
├── ml_tools.py           6 ML tools, registry, validating dispatcher, sklearn wrapper
├── react_agent.py        ReAct controller, Ollama client, self-correction, grounding audit
├── benchmark_runner.py   direct / agent / latency modes, paired-CV significance tests
├── run_traces.py         generates the six execution traces
├── setup_windows.ps1     WSL2 install (Administrator, once)
├── setup_wsl.sh          full in-WSL bootstrap (idempotent)
├── requirements.txt
├── tests/                109 offline tests (no Ollama or network required)
├── docs/                 generated results + engineering appendix
└── logs/                 execution traces: transcripts and structured records
```

---

# Part II — Technical Report

## 1. System architecture

```
                  WINDOWS 11 HOST
  ┌────────────────────────────────────────────────────────────┐
  │  NVIDIA driver (RTX 5060 Ti) ───┐                          │
  └─────────────────────────────────┼──────────────────────────┘
                  WSL2 (Ubuntu 22.04)│ GPU passthrough via /usr/lib/wsl/lib
  ┌─────────────────────────────────┼──────────────────────────┐
  │   ┌──────────────────┐  HTTP    ▼  ┌────────────────────┐  │
  │   │  react_agent.py  │ ───────────►│  Ollama daemon     │  │
  │   │  ReAct controller│  /api/      │  127.0.0.1:11434   │  │
  │   │                  │  generate   │  llama3.2:3b       │  │
  │   │ • prompt builder │ ◄───────────│  4-bit GGUF, 2 GB  │  │
  │   │ • output parser  │  completion └────────────────────┘  │
  │   │ • self-correction│                                     │
  │   │ • stall breaker  │                                     │
  │   │ • grounding audit│                                     │
  │   └────────┬─────────┘                                     │
  │            │ call_tool(name, kwargs)                       │
  │            ▼                                               │
  │   ┌────────────────────────────────────────────────────┐   │
  │   │           ml_tools.py — TOOL REGISTRY              │   │
  │   │  ① alias normalisation   dataset → dataset_name    │   │
  │   │  ② signature validation  reject unknown kwargs     │   │
  │   │  ③ required-arg check    reject incomplete calls   │   │
  │   │  ④ dispatch + exception → typed error envelope     │   │
  │   ├────────────────────────────────────────────────────┤   │
  │   │ load_dataset_summary    train_sklearn_model        │   │
  │   │ train_pytorch_mlp       tune_hyperparameters       │   │
  │   │ feature_selection       train_deep_classifier      │   │
  │   └──────────┬──────────────────────┬──────────────────┘   │
  │              ▼                      ▼                      │
  │      ┌──────────────┐      ┌──────────────┐                │
  │      │ Scikit-Learn │      │   PyTorch    │                │
  │      └──────────────┘      └──────────────┘                │
  └────────────────────────────────────────────────────────────┘
```

**The controller loop.** Each cycle prompts the model, parses one turn, and routes it:

```
   user task ──► build prompt (system + tool catalogue + scratchpad) ◄──────────┐
                            │                                                   │
                            ▼                                                   │
                  query LLM (stop = "Observation:")                             │
                            │                                                   │
        ┌───────────────────┼────────────────────┬──────────────────┐           │
        ▼                   ▼                    ▼                  ▼           │
  "Final Answer:"      Action + Input      unparseable        (none of these)    │
        │                   │                    │                              │
        ▼                   ▼                    ▼                              │
  GROUNDING AUDIT     loop-breaker          reformat nudge ─────────────────────►┤
  every number in       (call cache)         (no tool runs)                      │
  an Observation?           │                                                    │
    │        │              ▼                                                    │
  pass   fabricated    execute tool ──► error? ──► SELF-CORRECTION ──────────────┤
    │        └──────────────┐              │        (strategy + hint             │
    ▼                       │              ▼         + retry_with)               │
  DONE                      └──────────────► Observation ────────────────────────┘
                                                │
              3 unproductive steps, or budget spent
                                                ▼
                    FORCED FINALISATION: pre-fill "Final Answer:"
                    and enumerate the only numbers it may quote
```

The loop is bounded on four axes: total iterations, repairs per call signature,
cache-suppressed repeats, and consecutive unproductive steps. Any one alone is
insufficient — a model that cannot be stopped from repeating a successful call exhausts
its budget before concluding.

## 2. Local LLM architecture in WSL2

WSL2 runs a real Linux kernel in a lightweight VM. Three consequences matter: Ollama ships
Linux-first; CUDA reaches the guest through the **Windows** driver mounted at
`/usr/lib/wsl/lib` (installing a Linux NVIDIA driver inside Ubuntu breaks passthrough — a
common and costly mistake); and `llama.cpp` compiles without Windows patches. Verified
here: `nvidia-smi` sees the RTX 5060 Ti from inside Ubuntu and `torch.cuda.is_available()`
is `True` at compute capability `(12, 0)` — Blackwell, requiring the `cu128` wheel that
`setup_wsl.sh` selects automatically.

**Quantization.** `llama3.2:3b` ships as a 4-bit GGUF of 2.0 GB against ~6.4 GB in FP16.
Each block of weights maps to a low-bit integer with a shared scale,
$w \approx s \cdot q$ with $q \in \{0..15\}$ and $s = \max|w_{\text{block}}| / 15$. The
`Q4_K_M` scheme keeps attention-output and feed-forward down-projection tensors at higher
precision, which is why it degrades far less than uniform 4-bit. Practically, the model
fits in GPU memory alongside the PyTorch jobs the agent launches.

**Transport.** `POST /api/generate`, `stream: false`, `temperature: 0.1` (the task is
protocol compliance, not creativity), and `stop: ["Observation:"]` — the single most
important setting, without which the model writes its own fabricated observation and then
reasons over invented numbers. Ollama's `eval_count` and `eval_duration` counters are
recorded per step to produce §6.

## 3. Prompt engineering

The system prompt is **generated from the tool registry**, so the catalogue the model sees
can never drift from the code. Techniques that measurably changed behaviour, in order of
impact:

1. **A worked `Action Input` example per tool.** Small models pattern-match far better
   than they follow prose; this removed most argument-shape errors.
2. **An explicit anti-hallucination clause.** Necessary but, as §5 shows, *not sufficient*.
3. **Explicit yield instruction** ("Then STOP and wait"), reinforcing the stop sequence
   semantically.
4. **Error-handling rules stated in advance**, so repair instructions arrive familiar.
5. **Enumerating permitted values.** During forced finalisation the prompt lists every
   number observed in the run and forbids any other. This converts open-ended generation
   into selection from a fixed set, and works far better than any instruction not to
   invent numbers.

**Naming is prompt engineering.** The summary tool originally returned
`class_balance: {"0": 59, "1": 71, "2": 48}`; the model read those counts as *percentages*
and reported figures summing to 178%. Renaming the field to `samples_per_class`, with a
separate `class_proportions`, fixed it. No prompt instruction did. Field names are read as
documentation, and an ambiguous name is a prompting bug.

## 4. Machine learning tools

**Task 1.** `load_dataset_summary`, `train_sklearn_model` (decision tree, logistic
regression, random forest) and `train_pytorch_mlp`. Logistic regression is wrapped in a
`Pipeline` with `StandardScaler` — `breast_cancer` features span four orders of magnitude,
so an unscaled fit is a convergence problem, not a modelling result. PyTorch
standardisation uses **training statistics only**; computing the mean before splitting
leaks test information.

**Task 2.**

- `tune_hyperparameters` — Grid or Randomized search over an SVC (kernel SVM, 60
  candidates) or decision tree (108). The estimator sits inside a `Pipeline` with a scaler
  so the scaler refits on every CV fold; scaling once beforehand leaks fold statistics. A
  top-3 leaderboard is returned, not just the winner.
- `feature_selection` — PCA and `SequentialFeatureSelector`, each scored against an
  identical full-feature baseline so the accuracy cost of compression is measured rather
  than assumed. PCA maximises retained variance; SFS greedily optimises CV accuracy. They
  answer different questions and both are reported.
- `train_deep_classifier` — configurable `[Linear → BatchNorm → ReLU → Dropout] × N →
  Linear`, with cosine/step/plateau schedules, weight decay, and a choice of Adam or SGD.
  Cosine annealing follows
  $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos(\tfrac{t}{T}\pi))$.
  It reports `generalisation_gap` = train − test accuracy, the quantity Dropout and
  BatchNorm exist to control.

All splits use `random_state=42` with stratification; every stochastic tool re-seeds on
entry. `TorchMLPClassifier` wraps the network in the Scikit-Learn estimator API, which is
what makes the paired comparison in §5 possible.

## 5. Self-correction (Task 3)

Tools never raise for recoverable problems. They return a typed envelope:

```json
{"status": "error", "error_type": "shape_mismatch",
 "message": "n_components=25 is invalid for iris, which has only 4 features.",
 "hint": "n_components must satisfy 1 <= n_components <= 4.",
 "retry_with": {"dataset_name": "iris", "method": "pca", "n_components": 4}}
```

Nine error types are mapped to corrective instructions. The design choice worth defending
is **`retry_with`**: handing back a corrected argument dictionary rather than only a
description collapses the model's job from *diagnose and re-derive* to *copy*, which is
what makes recovery reliable at 3B scale.

Three bounds stop recovery becoming its own failure mode: a three-attempt repair budget per
call signature, a cache that intercepts repeated successful calls (and names the tools not
yet used), and forced finalisation after three unproductive steps.

**The grounding audit.** The most dangerous failure produced no error at all. Asked to
benchmark six model/dataset combinations, the agent ran one, then emitted a confident,
well-formatted six-row results table in which every figure was invented — including a
clinical-screening recommendation. The prompt's "never invent numerical results"
instruction did not prevent it. `audit_grounding()` now checks every decimal in a Final
Answer against the numbers that actually appeared in observations, accepting rounding
(0.96 from 0.9561) and percentage rendering (97.42 from 0.9742). Fabricated answers are
pushed back into the loop; a fabricated forced answer is regenerated once, then refused
outright rather than reported.

**Observed behaviour.** Six scenarios were run against both `llama3.2:3b` and `mistral:7b`
— twelve traces. **All twelve reach a Final Answer and all twelve pass the grounding
audit**, every reported number traced to an observation.

| Scenario | Demonstrates | 3B steps/corrections | 7B steps/corrections |
|---|---|---|---|
| `single_tool` | Baseline ReAct cycle | 6 / 0 | 6 / 0 |
| `multi_tool` | Chaining three tools | 10 / 2 | 4 / 0 |
| `self_correction` | **Shape mismatch** recovery | 7 / 1 | 3 / 1 |
| `nan_recovery` | **NaN loss** recovery | 8 / 1 | 3 / 1 |
| `parameter_error` | **Parameter error** recovery | 6 / 0 | 6 / 4 |
| `full_pipeline` | Tuning + PCA + deep net | 8 / 0 | 4 / 0 |

`nan_recovery` is the cleanest demonstration: both models request `lr=9.9`, receive
`nan_loss`, read the suggested value, retry at `lr=0.1`, and reach 0.9474 test accuracy.
The difference between models is not success versus failure but *directness* — 45 steps
versus 26 across the six scenarios.

## 6. Experimental results and statistical analysis

### 6.1 Cross-validated comparison

`benchmark_runner.py --mode direct` evaluates three algorithms across two datasets. All
three are scored on the **same 10 stratified folds**, with the SVM's hyperparameter search
nested *inside* each outer training fold so model selection never sees the fold it is
scored on.

| Dataset | Algorithm | Mean accuracy | Std | Per-fold range |
|---|---|---|---|---|
| wine | **Random Forest** | **0.9889** | 0.0222 | 0.9444 – 1.0000 |
| wine | Kernel SVM (tuned) | 0.9833 | 0.0255 | 0.9444 – 1.0000 |
| wine | Deep MLP (Dropout+BN) | 0.9830 | 0.0260 | 0.9412 – 1.0000 |
| breast_cancer | **Kernel SVM (tuned)** | **0.9772** | 0.0193 | 0.9474 – 1.0000 |
| breast_cancer | Deep MLP (Dropout+BN) | 0.9701 | 0.0176 | 0.9474 – 1.0000 |
| breast_cancer | Random Forest | 0.9526 | 0.0314 | 0.8772 – 0.9825 |

**Optimism bias is measurable.** Under the simpler protocol — tuning and scoring on the
same folds — the wine SVM reports 0.9931 and looks like a clear winner. Nested CV drops it
to 0.9833 and the Random Forest overtakes it. That 0.0098 gap is the price of selecting
hyperparameters on the evaluation folds, and it is invisible unless the comparison is done
properly.

### 6.2 Are the differences significant? (CO2)

Differences are tested with the **Nadeau–Bengio corrected resampled t-test**, which
Raschka recommends for this situation. The ordinary paired t-test assumes independent
folds; CV folds share training data, so the naive test understates variance and
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

**None of the six comparisons is significant.** The defensible conclusion is that all three
algorithms are statistically indistinguishable on both datasets at these sample sizes. On a
representative pair the corrected test returns p = 0.0036 where the uncorrected returns
p = 0.0003 — an order of magnitude more conservative, and the reason "significant"
differences in small-data ML so often fail to replicate.

### 6.3 Interpretation (CO1)

On tabular problems of a few hundred rows, **a tuned kernel SVM and a random forest match
a regularised deep network at a fraction of the training cost.** The network needs
3,267–4,322 parameters to fit datasets of 178–569 samples: its capacity is not the binding
constraint, sample size is. The measured generalisation gaps (0.033 wine, 0.022
breast_cancer) show Dropout and BatchNorm doing their job, but controlling over-fitting
cannot manufacture information the data does not contain. This mirrors the textbook's
framing — model selection is bounded by the bias–variance trade-off and the available data,
not by architectural sophistication.

Note also the wine random forest scoring a *perfect* 1.0000 on the hold-out test set while
ranking second on cross-validation. With a 36-row test set, 1.0000 and 0.9722 differ by one
sample: precisely why the cross-validated mean is the comparison used throughout.

## 7. Inference latency in WSL2

`benchmark_runner.py --mode latency --repeats 5`, on an RTX 5060 Ti with 15 GB allocated to
WSL2:

| Prompt type | Mean (s) | Std (s) | Min (s) | Max (s) | Tokens | Tok/s |
|---|---|---|---|---|---|---|
| short | 0.672 | 1.302 | 0.016 | 3.277 | 2.8 | 231.04 |
| medium | 0.744 | 0.044 | 0.686 | 0.797 | 120 | 167.24 |
| react_turn | 0.119 | 0.015 | 0.108 | 0.148 | 17.6 | 173.29 |

The `short` row's standard deviation exceeds its mean — an artefact worth reading
correctly. The first call includes a one-time 3.277 s model load; the remaining four run in
0.016 s. Quoting "0.672 s mean" would mislead. The steady-state figure is **167–174 tok/s
with GPU offload**, roughly 10× CPU-only inference for a 4-bit 3B model.

**Where wall-clock time goes.** Tool execution dominates: in `full_pipeline`, 7 LLM calls
consumed 4.7 s against 59.5 s of tool time, since a 60-candidate grid search is simply
expensive. But inference cost grows with every step, because the scratchpad is re-sent in
full — the direct argument for the loop-breaker and repair budget in §5, each suppressed
redundant call saving an entire forward pass over a monotonically growing prompt.

## 8. Verification

**109 tests, no Ollama and no network required.** A `ScriptedLLM` replays fixed model
turns, making behaviours that would otherwise need a live model deterministic:
`test_agent_self_corrects_a_shape_mismatch` asserts not merely that the run finishes but
that the exact `retry_with` payload reached the model on the following turn;
`test_fabricated_final_answer_is_rejected_and_sent_back_to_the_loop` reproduces the
fabrication in §5 and asserts the recovery.

| File | Tests | Covers |
|---|---|---|
| `tests/test_ml_tools.py` | 48 | All six tools, all datasets, every scheduler, determinism, every error path |
| `tests/test_react_agent.py` | 52 | Parsing, tool chaining, self-correction, stall-breaking, grounding audit, logging |
| `tests/test_statistics.py` | 9 | Corrected resampled t-test, paired CV, significance rendering |

Nine defects in this project produced plausible output and were **silently wrong** — none
raised an exception. Among them: a significance test that inverted its own verdict on
zero-variance differences; a grounding audit that let fabrications validate themselves;
`bool("false")` evaluating True, so `batch_norm=false` quietly trained *with* BatchNorm;
and an optional colon in the `Action` header that let the prose "…retry the same Action
with…" dispatch a tool named `with`. The full log is in
[`docs/APPENDIX.md`](docs/APPENDIX.md) §10. That class of bug is the argument for running
the real model early rather than after the code looks finished.

**Environment:** Ubuntu 22.04.5 LTS on WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2),
Python 3.10.12, PyTorch 2.11.0+cu128 with CUDA available, NVIDIA RTX 5060 Ti (sm_120),
Ollama serving `llama3.2:3b` (4-bit GGUF, 2.0 GB) and `mistral:7b`. ML results are
seed-fixed and reproduce on re-run; latency is hardware-dependent.

## 9. Limitations and future work

- **Model scale is the binding constraint, and it is measured rather than asserted.** On
  the six-experiment benchmark prompt, `llama3.2:3b` completes 1 of 6 experiments and
  invents the rest; `mistral:7b` completes 6 of 6 with 17 of 17 numbers grounded
  (`docs/BENCHMARK_AGENT.md`). The reliability scaffolding lets a 3B model fail *safely*;
  it does not make it capable.
- **The scratchpad grows without bound.** Each step re-sends the whole history. Beyond a
  dozen steps this dominates latency; summarising older observations, or using Ollama's
  `context` token array, would bound it.
- **Grounding is checked numerically, not semantically.** The audit confirms a number
  appeared in *some* observation, not that it was attached to the right model or dataset.
  An answer swapping two correctly-observed accuracies would pass.
- **Three datasets, all small and clean.** No missing values, no categorical encoding, no
  real class imbalance.
- **Constrained decoding is unused.** Ollama supports grammar- and JSON-schema-constrained
  generation, which would eliminate the parsing failure modes outright rather than
  recovering from them.

---

## Reproducing every artefact in this report

```bash
bash setup_wsl.sh && source venv/bin/activate
python -m pytest tests -q                                  # 109 tests, offline
python run_traces.py --tag llama32-3b                      # §5 traces
python run_traces.py --model mistral:7b --tag mistral7b    # §5, §9 comparison
python benchmark_runner.py --mode direct                   # §6 tables
python benchmark_runner.py --mode latency --repeats 5      # §7 table
python benchmark_runner.py --mode agent --model mistral:7b # Task 3 benchmark
```
