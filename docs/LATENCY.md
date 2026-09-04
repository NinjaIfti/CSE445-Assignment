# Local LLM Latency Benchmark (WSL2)

Model: `llama3.2:3b` served by Ollama. Measured 2026-09-04 14:10:07, 5 repeats per prompt type.

| Prompt type | Mean (s) | Std (s) | Min (s) | Max (s) | Tokens generated | Tok/s |
|---|---|---|---|---|---|---|
| short | 0.672 | 1.302 | 0.016 | 3.277 | 2.8 | 231.04 |
| medium | 0.744 | 0.044 | 0.686 | 0.797 | 120 | 167.24 |
| react_turn | 0.119 | 0.015 | 0.108 | 0.148 | 17.6 | 173.29 |

Raw measurements:

```json
[
  {
    "prompt_type": "short",
    "runs": 5,
    "mean_seconds": 0.672,
    "std_seconds": 1.302,
    "min_seconds": 0.016,
    "max_seconds": 3.277,
    "mean_tokens_generated": 2.8,
    "mean_tokens_per_second": 231.04
  },
  {
    "prompt_type": "medium",
    "runs": 5,
    "mean_seconds": 0.744,
    "std_seconds": 0.044,
    "min_seconds": 0.686,
    "max_seconds": 0.797,
    "mean_tokens_generated": 120,
    "mean_tokens_per_second": 167.24
  },
  {
    "prompt_type": "react_turn",
    "runs": 5,
    "mean_seconds": 0.119,
    "std_seconds": 0.015,
    "min_seconds": 0.108,
    "max_seconds": 0.148,
    "mean_tokens_generated": 17.6,
    "mean_tokens_per_second": 173.29
  }
]
```
