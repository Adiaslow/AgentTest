#!/usr/bin/env bash
# Bootstrap for macOS and Linux: installs uv if missing, then runs setup.py.
set -e

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "→ Installing uv (https://docs.astral.sh/uv/)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin by default; make it available now
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "✗ uv install succeeded but 'uv' is not on PATH." >&2
        echo "  Open a new terminal (so your shell rc picks up ~/.local/bin) and re-run ./setup.sh" >&2
        exit 1
    fi
fi

exec uv run --no-project setup.py
