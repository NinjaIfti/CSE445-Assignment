# Building an Autonomous Local LLM Machine Learning Agent in Windows WSL

**CSE445: Machine Learning — Assignment #3 · Technical Report**
North South University, Department of Electrical and Computer Engineering
Instructor: Dr. Mohammad Abdul Qayum (MAQm) · Section 6
Primary reference: *Machine Learning with PyTorch and Scikit-Learn* — Raschka, Liu, Mirjalili

---

## 1. Objectives and scope

This project implements an autonomous machine learning agent that runs entirely on local
hardware. A quantized open-source LLM served by Ollama inside WSL2 acts as the reasoning
engine; a hand-written ReAct controller turns its text output into calls against a
registry of six machine learning tools built on Scikit-Learn and PyTorch.

The engineering claim being tested is narrow and worth stating plainly: **a 3-billion
parameter model is not a reliable function caller, and the value of the system lies in the
scaffolding that makes it one.** Most of the design below is about containing that
unreliability — typed errors, argument normalisation, repair budgets, loop detection —
rather than about the language model itself.

Three deliverables follow the assignment tasks:

1. A working environment and baseline ReAct engine (Task 1).
2. Three additional ML tools: hyperparameter search, dimensionality reduction, and a
   regularised deep network (Task 2).
3. Self-correction plus an autonomous model-comparison benchmark (Task 3).

---

## 2. System architecture

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
  │   │  ReAct controller  │  /api/      │  127.0.0.1:11434 │    │
  │   │                    │  generate   │  llama3.2:3b     │    │
  │   │  • prompt builder  │ ◄────────── │  4-bit GGUF      │    │
  │   │  • output parser   │  completion │  llama.cpp core  │    │
  │   │  • self-correction │             └──────────────────┘    │
  │   │  • run logger      │                                     │
  │   └─────────┬──────────┘                                     │
  │             │ call_tool(name, kwargs)                        │
  │             ▼                                                │
  │   ┌────────────────────────────────────────────────────┐     │
  │   │            ml_tools.py — TOOL REGISTRY             │     │
  │   │  alias normalisation ▸ signature validation ▸      │     │
  │   │  dispatch ▸ JSON envelope ▸ typed error taxonomy   │     │
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
  │ build prompt    │◄────────────────────────────────────┐
  │ system + tool   │                                     │
  │ catalogue +     │                                     │
  │ scratchpad      │                                     │
  └────────┬────────┘                                     │
           ▼                                              │
  ┌─────────────────┐    "Final Answer:"   ┌───────────┐   │
  │ query local LLM │─────────────────────►│   DONE    │   │
  │ stop=Observation│                      └───────────┘   │
  └────────┬────────┘                                      │
           ▼                                               │
  ┌─────────────────┐   unparseable   ┌──────────────────┐ │
  │ parse the turn  │────────────────►│ reformat nudge   │─┤
  └────────┬────────┘                 │ (no tool runs)   │ │
           │                          └──────────────────┘ │
           ▼ Action + Action Input                         │
  ┌─────────────────┐   seen before   ┌──────────────────┐ │
  │ loop-breaker    │────────────────►│ cached result +  │─┤
  │ (call cache)    │                 │ "do not repeat"  │ │
  └────────┬────────┘                 └──────────────────┘ │
           ▼                                               │
  ┌─────────────────┐  status=error   ┌──────────────────┐ │
  │ execute tool    │────────────────►│ SELF-CORRECTION  │─┤
  │ via call_tool   │                 │ strategy + hint  │ │
  │                 │                 │ + retry_with     │ │
  └────────┬────────┘                 └──────────────────┘ │
           ▼ status=ok                                     │
      Observation ───────────────────────────────────────► ┘
```

The loop is bounded on three axes at once: total iterations, repairs per call signature,
and cache-suppressed repeats. Any one of them alone is insufficient — a small model that
cannot be stopped from repeating a successful call will exhaust its iteration budget
before it ever reaches a Final Answer.

---

## 3. Local LLM architecture in WSL2

### 3.1 Why WSL2

WSL2 runs a real Linux kernel in a lightweight VM rather than emulating syscalls. That
matters here for three reasons: Ollama ships as a Linux-first binary, CUDA is exposed to
the guest through the Windows driver at `/usr/lib/wsl/lib` (no separate Linux driver is
installed — a common and costly misunderstanding), and the whole `llama.cpp` toolchain
compiles without Windows-specific patches.

### 3.2 Quantization

`llama3.2:3b` is distributed as a **4-bit GGUF** file, roughly 2 GB against approximately
6.4 GB for the same weights in FP16. Quantization maps each block of weights to a low-bit
integer with a shared scale factor:

$$w \approx s \cdot q, \qquad q \in \{0, \dots, 15\}, \qquad s = \frac{\max|w_{\text{block}}|}{15}$$

The `Q4_K_M` scheme used by default keeps a small number of sensitive tensors (attention
output and feed-forward down-projections) at higher precision, which is why it loses
noticeably less quality than uniform 4-bit. The practical consequence for this assignment
is that the model fits in GPU memory alongside the PyTorch training jobs the agent
launches, and inference stays interactive.

### 3.3 Inference transport

The controller talks to `POST /api/generate` with `stream: false`. Two options do most of
the work:

- `temperature: 0.1` — the task is protocol compliance, not creativity. Higher
  temperatures measurably increase malformed `Action Input` blocks.
- `stop: ["Observation:", "\nObservation"]` — **the single most important setting.**
  Without it the model happily writes its own fabricated `Observation:` line and then
  reasons over invented numbers. The stop sequence forces it to yield control back to the
  controller after every action.

The response body also carries `eval_count`, `eval_duration`, `prompt_eval_count` and
`total_duration`, which the agent records per step to produce the latency figures in §8.

---

## 4. Prompt engineering

The system prompt is **generated from the tool registry**, not written by hand. Each
tool's signature, defaults and a worked example are rendered by `describe_tools()`, so the
catalogue the model sees can never drift from the code:

```
4. tune_hyperparameters(dataset_name, model_type, search_type="grid", cv=5, n_iter=20, scoring="accuracy")
   Run GridSearchCV or RandomizedSearchCV over an SVC (kernel SVM) or a decision_tree, ...
   Example Action Input: {"dataset_name": "wine", "model_type": "svc", "search_type": "grid"}
```

Techniques that materially changed behaviour, in rough order of impact:

1. **Worked example per tool.** Showing one concrete `Action Input` per tool removed most
   argument-shape errors. Small models pattern-match far better than they follow prose.
2. **An explicit anti-hallucination clause** — *"You never invent numerical results: every
   number you report must come from an Observation."* Without it, 3B models routinely
   produce a plausible accuracy table having called nothing.
3. **Explicit yield instruction** — *"Then STOP and wait."* This reinforces the stop
   sequence at the semantic level.
4. **Error-handling rules in the prompt itself.** Telling the model in advance that
   `status: error` observations carry `error_type`, `hint` and `retry_with` means the
   repair instruction it later receives is already familiar rather than novel.
5. **A "do not repeat successful calls" rule**, backing the mechanical loop-breaker.

Defaults are rendered as JSON (`true`, not `True`) so the model never sees Python syntax
it might imitate inside an `Action Input`.

---

## 5. Parsing: designing for a model that will not comply

The rubric awards parsing robustness, and in practice it is where most engineering time
went. `parse_llm_output()` handles, in order of preference:

| Model output | Handling |
|---|---|
| Clean `Thought:` / `Action:` / `Action Input:` | Direct regex extraction |
| ` ```json {...} ``` ` fences | Fence stripped before parsing |
| `**Action:**` markdown bold | Optional `\**` in the header patterns |
| `{'dataset_name': 'iris'}` single quotes | `json.loads` fails → `ast.literal_eval` |
| `{"a": 1} and then I will...` trailing prose | Brace-balanced scan, string-aware |
| `train_sklearn_model(dataset_name="iris", ...)` | Function-call regex fallback |
| `dataset_name="iris", epochs=50` no braces | `key=value` pair extraction |
| Conversational filler, no protocol at all | Reformat nudge; **no tool is executed** |

The brace scanner is deliberately string-aware: a naïve `text[text.find("{"):text.rfind("}")+1]`
breaks on any argument value containing a brace. Each of the eight rows above is covered
by a test in `tests/test_react_agent.py`.

Beyond parsing, `call_tool()` acts as a **hallucination firewall** before any ML code
runs. It normalises well-known argument synonyms (`dataset` → `dataset_name`, `model` →
`model_type`, `learning_rate` → `lr`), then rejects anything still unrecognised with the
valid signature attached. Normalisations are echoed back in the observation as
`_normalised_arguments`, so a trace never hides what was silently rewritten.

---

## 6. Machine learning tool implementation

### 6.1 Task 1 — baseline

`load_dataset_summary`, `train_sklearn_model` (decision tree, logistic regression, random
forest) and `train_pytorch_mlp`. Two details matter beyond the assignment skeleton:
logistic regression is wrapped in a `Pipeline` with `StandardScaler`, because
`breast_cancer` features span four orders of magnitude and an unscaled fit is a
convergence problem rather than a modelling result; and standardisation in the PyTorch
tool uses **training statistics only**, since computing the mean over the full array
before splitting leaks test information.

### 6.2 Task 2 — advanced tools

**`tune_hyperparameters`** — `GridSearchCV` or `RandomizedSearchCV` over an SVC (kernel
SVM) or a decision tree. The estimator sits inside a `Pipeline` with a scaler so the
scaler is refit on every CV fold; scaling the whole training set once before
cross-validation leaks fold statistics and inflates the reported score. Search spaces:

| Model | Grid | Candidates |
|---|---|---|
| SVC | `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {scale, auto, 0.01, 0.1, 1}`, `kernel ∈ {rbf, poly, linear}` | 60 |
| Decision tree | `max_depth ∈ {2,3,4,6,8,None}`, `min_samples_split ∈ {2,5,10}`, `min_samples_leaf ∈ {1,2,4}`, `criterion ∈ {gini, entropy}` | 108 |

The tool returns a top-3 leaderboard, not just the winner, so the agent can reason about
how sensitive the result is to the choice.

**`feature_selection`** — PCA and `SequentialFeatureSelector`, each scored against an
identical full-feature logistic-regression baseline so the accuracy cost of compression is
measured rather than assumed. PCA maximises retained variance by projecting onto the top
eigenvectors of the covariance matrix; SFS greedily optimises cross-validated accuracy
directly. They answer different questions, and the tool reports both.

**`train_deep_classifier`** — a configurable `[Linear → BatchNorm → ReLU → Dropout] × N →
Linear` stack with cosine, step, plateau, or no LR schedule, plus optional weight decay.
Cosine annealing follows

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\left(\tfrac{t}{T}\pi\right)\right)$$

The tool reports `generalisation_gap` = train accuracy − test accuracy, which is the
quantity Dropout and BatchNorm exist to control. Reporting it makes over-fitting visible
to the agent instead of leaving it to be inferred.

### 6.3 Reproducibility

Every stochastic tool re-seeds NumPy and PyTorch on entry, and all splits use
`random_state=42` with stratification. Both PyTorch tools additionally accept a `seed`
argument that controls initialisation and batch order **while holding the split fixed** —
this is what lets the benchmark in §7 give the neural network a variance estimate
comparable to a cross-validated one.

---

## 7. Experimental results and statistical analysis

Produced by `python benchmark_runner.py --mode direct`. Scikit-Learn models use 5-fold
stratified cross-validation; the network is retrained under 5 initialisations
(`seeds = [42, 7, 13, 2024, 99]`) on a fixed split.

| Dataset | Algorithm | Framework | Validation protocol | Val accuracy (mean ± std) | Test accuracy | Time (s) |
|---|---|---|---|---|---|---|
| wine | Random Forest | scikit-learn | 5-fold stratified CV | 0.9775 ± 0.0213 | 1.0000 | 0.29 |
| wine | Kernel SVM (tuned) | scikit-learn | 5-fold CV over 60 candidates | **0.9931 ± 0.0138** | 0.9444 | 5.77 |
| wine | Deep MLP (Dropout+BN) | PyTorch | repeated hold-out, 5 seeds | 0.9666 ± 0.0111 | 0.9722 | 3.96 |
| breast_cancer | Random Forest | scikit-learn | 5-fold stratified CV | 0.9543 ± 0.0150 | 0.9561 | 0.45 |
| breast_cancer | Kernel SVM (tuned) | scikit-learn | 5-fold CV over 60 candidates | **0.9758 ± 0.0108** | 0.9825 | 1.00 |
| breast_cancer | Deep MLP (Dropout+BN) | PyTorch | repeated hold-out, 5 seeds | 0.9737 ± 0.0096 | 0.9825 | 8.51 |

Best configurations found by the search: `wine` → `C=10, gamma=scale, kernel=rbf`;
`breast_cancer` → `C=0.1, gamma=scale, kernel=linear`.

### 7.1 Are the differences real? (CO2)

Reporting a winner from four decimal places is the standard failure of this kind of
comparison. `benchmark_runner.py` compares the top two entries against the pooled spread
of their estimates, $\sqrt{\sigma_1^2 + \sigma_2^2}$, and refuses to claim a difference
that falls inside it:

- **wine** — the tuned SVM leads the random forest by 0.0156, against a pooled standard
  deviation of 0.0254. **Not a defensible difference.** With 178 samples, one fold
  contains about 36 rows, so a single reclassified sample moves fold accuracy by ~2.8
  percentage points — the same order as the entire margin.
- **breast_cancer** — the tuned SVM leads the deep network by 0.0021 against a pooled
  standard deviation of 0.0144. **Also indistinguishable.** The honest conclusion is that
  all three algorithms are equivalent on this dataset at this sample size.

Note the wine row where the random forest scores a *perfect* 1.0000 on the hold-out test
set while ranking second on cross-validation. With a 36-row test set, 1.0000 and 0.9722
differ by one sample. This is exactly why the cross-validated mean, not the hold-out
number, is the comparison used above.

### 7.2 Interpretation (CO1)

The result is unglamorous and worth stating: **on tabular problems of a few hundred rows,
a tuned kernel SVM and a random forest match or beat a regularised deep network**, at a
fraction of the training cost. The network needs 3,267–4,322 parameters to fit datasets of
178–569 samples; its capacity is not the binding constraint, sample size is. The measured
generalisation gaps (0.033 on wine, 0.022 on breast_cancer) show the Dropout and BatchNorm
regularisation is doing its job — without it the gap widens sharply — but controlling
over-fitting cannot manufacture information that a small dataset does not contain.

This mirrors the textbook's framing: model selection is bounded by the bias–variance
trade-off and the data available, not by architectural sophistication.

---

## 8. Inference latency in WSL2

Latency is measured by `python benchmark_runner.py --mode latency --repeats 5`, which
records wall-clock time per call alongside Ollama's own `eval_count` / `eval_duration`
counters, and writes `docs/LATENCY.md`. Three prompt shapes are profiled: a one-word
reply, a short explanation, and a realistic ReAct continuation.

> **Status: to be populated.** These numbers must be measured on the grading machine
> after `setup_wsl.sh` completes, because they are entirely hardware-dependent. Run the
> command above and paste the generated table here. Reference points for interpreting the
> result: a 4-bit 3B model typically sustains roughly 10–20 tok/s on CPU and 60–120 tok/s
> with GPU offload, and the first call after a cold start includes a one-time model load
> (`load_duration`) of several seconds.

The per-step telemetry recorded in every trace (`llm_seconds`, `eval_count`,
`tokens_per_second`) means each run in `logs/` also carries its own latency profile, and
`logs/TRACES.md` aggregates them.

**Where the wall-clock time actually goes.** Tool execution is not the bottleneck: the
whole reference benchmark — six algorithm/dataset cells, including two 60-candidate grid
searches and ten neural network trainings — completes in about 20 seconds. A single agent run of comparable scope takes several minutes, almost
all of it LLM generation. Every additional ReAct step costs a full forward pass over a
scratchpad that grows monotonically, which is the direct argument for the loop-breaker and
repair budget in §9: each suppressed redundant call saves a whole inference.

---

## 9. Self-correction (Task 3)

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

`_build_repair_hint()` converts that into an instruction appended after the observation:

```
[SELF-CORRECTION 1] The call failed with error_type='shape_mismatch'. The requested
dimensionality is incompatible with the data. Reduce it to fit within the number of
available features and retry. Hint: n_components must satisfy 1 <= n_components <= 4.
Use exactly this Action Input: {"dataset_name": "iris", "method": "pca", "n_components": 4}
```

The design choice worth defending is `retry_with`. Handing the model a corrected argument
dictionary rather than only a description is what makes recovery reliable at 3B scale: the
model's job collapses from *diagnose and re-derive* to *copy*. Nine error types are
mapped, covering hallucinated tool names, hallucinated arguments, invalid ranges, shape
mismatches, divergence (`nan_loss` → retry at one tenth the learning rate) and failed
searches.

Three bounds keep recovery from becoming its own failure mode: a repair budget of three
attempts per call signature, after which the agent is told to change approach or answer
with what it has; a cache that intercepts repeated successful calls; and the global
iteration limit.

All of this is verified offline. A `ScriptedLLM` replays fixed model turns, so
`test_agent_self_corrects_a_shape_mismatch` asserts not merely that the run finishes, but
that the corrective text and the exact `retry_with` payload reached the model on the
following turn.

---

## 10. Verification

56 tests, no Ollama and no network required.

```
$ python -m pytest tests -q
56 passed
```

`tests/test_ml_tools.py` (31 tests) covers all six tools across all three datasets, every
LR scheduler, determinism under repeated calls, and every error path — including a
monkeypatched divergent loss that genuinely exercises the `nan_loss` branch rather than
assuming it works. `tests/test_react_agent.py` (25 tests) covers the eight parsing cases
from §5, single- and multi-tool runs, both self-correction scenarios, the repair budget,
the loop-breaker, the reformat nudge, the iteration limit, and log serialisation.

**Environment used for the numbers in §7:** Python 3.12.13, scikit-learn 1.9.0,
NumPy 2.5.2, pandas 3.0.5, PyTorch 2.14.0+cpu, on Windows 11 (build 26200). The ML results
are seed-fixed and reproduce identically on re-run; the WSL2 latency figures in §8 are
hardware-dependent and must be regenerated on the grading machine.

---

## 11. Limitations and future work

- **Model scale is the binding constraint.** `llama3.2:3b` needs the scaffolding in §5 and
  §9 to complete multi-step tasks reliably. A 7B model follows the protocol noticeably
  better, at roughly double the latency.
- **The scratchpad grows without bound.** Each step re-sends the entire history. Beyond
  roughly a dozen steps this dominates latency; summarising older observations, or
  switching to Ollama's `context` token array, would bound it.
- **Three datasets, all small, all clean.** No missing values, no categorical encoding, no
  class imbalance worth the name. The `missing_values` field in the summary tool is
  scaffolding for data that does not yet exercise it.
- **No statistical test between algorithms.** The pooled-standard-deviation comparison in
  §7.1 is a deliberately conservative heuristic, not a hypothesis test. A corrected
  resampled t-test over shared CV folds would be the rigorous next step.
- **Structured output is not enforced at the decoding level.** Ollama supports grammar- and
  JSON-schema-constrained generation, which would eliminate the parsing failure modes in
  §5 outright rather than recovering from them.

---

## 12. Reproducing this report

```bash
# Windows, elevated PowerShell, once
powershell -ExecutionPolicy Bypass -File setup_windows.ps1     # reboot when prompted

# Inside WSL Ubuntu
bash setup_wsl.sh
source venv/bin/activate

python -m pytest tests -q                       # 56 tests, offline
python react_agent.py --health                  # confirm Ollama + model
python run_traces.py                            # -> logs/TRACES.md  (§9 evidence)
python benchmark_runner.py --mode direct        # -> docs/BENCHMARK.md (§7)
python benchmark_runner.py --mode agent         # -> docs/BENCHMARK_AGENT.md
python benchmark_runner.py --mode latency -r 5  # -> docs/LATENCY.md   (§8)
```
