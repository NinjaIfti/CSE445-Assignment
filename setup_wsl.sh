#!/usr/bin/env bash
# ======================================================================================
# setup_wsl.sh -- one-shot environment bootstrap for CSE445 Assignment 3.
#
# Run this INSIDE a WSL2 Ubuntu shell, from the repository directory:
#     bash setup_wsl.sh
#
# It is idempotent: re-running it skips anything already installed.
#
# Options:
#     --cpu              force the CPU PyTorch wheel even if an NVIDIA GPU is visible
#     --cuda <ver>       CUDA wheel tag to use (default: auto -> cu128, else cpu)
#     --model <name>     Ollama model to pull (default: llama3.2:3b)
#     --skip-tests       do not run the offline test suite at the end
# ======================================================================================
set -euo pipefail

MODEL="llama3.2:3b"
CUDA_TAG=""
FORCE_CPU=0
RUN_TESTS=1
VENV_DIR="venv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)        FORCE_CPU=1; shift ;;
        --cuda)       CUDA_TAG="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --skip-tests) RUN_TESTS=0; shift ;;
        -h|--help)    sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }

# --------------------------------------------------------------------------------------
say "0/6  Verifying we are inside WSL2"
# --------------------------------------------------------------------------------------
if ! grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null; then
    warn "This does not look like WSL. The script still works on native Linux."
else
    ok "WSL detected: $(grep -oiE 'microsoft[^ ]*' /proc/version | head -1)"
fi
echo "Distro : $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "Kernel : $(uname -r)"

# --------------------------------------------------------------------------------------
say "1/6  Installing system packages"
# --------------------------------------------------------------------------------------
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv curl build-essential git
ok "python3 $(python3 --version | cut -d' ' -f2)"

# --------------------------------------------------------------------------------------
say "2/6  Installing Ollama"
# --------------------------------------------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
    ok "Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
fi

# --------------------------------------------------------------------------------------
say "3/6  Starting the Ollama daemon"
# --------------------------------------------------------------------------------------
if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    ok "Daemon already serving on port 11434"
else
    echo "Launching 'ollama serve' in the background (log: /tmp/ollama.log)"
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        ok "Daemon is up"
    else
        warn "Daemon did not come up in 30s. Check /tmp/ollama.log"
        exit 1
    fi
fi

# --------------------------------------------------------------------------------------
say "4/6  Pulling the quantized model: $MODEL"
# --------------------------------------------------------------------------------------
if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$MODEL"; then
    ok "$MODEL already pulled"
else
    ollama pull "$MODEL"
    ok "$MODEL pulled"
fi
echo "Models available:"
ollama list

# --------------------------------------------------------------------------------------
say "5/6  Creating the Python virtual environment"
# --------------------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    ok "venv created at ./$VENV_DIR"
else
    ok "venv already exists at ./$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip

# Decide which PyTorch wheel to fetch.
if [[ $FORCE_CPU -eq 1 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "PyTorch: CPU build (forced with --cpu)"
elif [[ -n "$CUDA_TAG" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
    echo "PyTorch: CUDA build ${CUDA_TAG} (explicit)"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    # WSL2 exposes the Windows driver through /usr/lib/wsl/lib. cu128 covers Ada and
    # Blackwell cards (RTX 40xx / 50xx); pass --cuda cu121 for older GPUs.
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    echo "PyTorch: CUDA build cu128 -- GPU detected:"
    nvidia-smi -L
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "PyTorch: CPU build (no NVIDIA GPU visible to WSL)"
fi

python -c "import torch" 2>/dev/null && ok "torch already installed" || \
    pip install torch --index-url "$TORCH_INDEX"

pip install --quiet -r requirements.txt
ok "Python dependencies installed"

python - <<'PYCHECK'
import sklearn, numpy, pandas, torch
print(f"  scikit-learn {sklearn.__version__}")
print(f"  numpy        {numpy.__version__}")
print(f"  pandas       {pandas.__version__}")
print(f"  torch        {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU          {torch.cuda.get_device_name(0)}")
PYCHECK

# --------------------------------------------------------------------------------------
say "6/6  Verifying the installation"
# --------------------------------------------------------------------------------------
python react_agent.py --health --model "$MODEL" || warn "Agent health check failed"

if [[ $RUN_TESTS -eq 1 ]]; then
    echo
    echo "Running the offline test suite..."
    python -m pytest tests -q
fi

cat <<EOF

======================================================================
Setup complete.

  source $VENV_DIR/bin/activate

Next steps:
  python react_agent.py --health              # confirm Ollama + model
  python run_traces.py                        # generate the 3 required traces
  python benchmark_runner.py --mode direct    # reference benchmark table
  python benchmark_runner.py --mode agent     # let the agent run the benchmark
  python benchmark_runner.py --mode latency   # WSL2 inference latency numbers
======================================================================
EOF
