# Philosopher Agents

Multi-agent philosophical discussions powered by a local LLM (via [Ollama](https://ollama.com)), with optional text-to-speech so the philosophers can actually talk.

## Quickstart

Works on macOS, Linux, and Windows. Clone the repo and run one command.

**macOS / Linux:**
```bash
git clone <this-repo> philosopher-agents
cd philosopher-agents
./setup.sh
uv run philosopher_agents.py
```

**Windows (PowerShell):**
```powershell
git clone <this-repo> philosopher-agents
cd philosopher-agents
.\setup.ps1
uv run philosopher_agents.py
```

The setup script installs everything you need, prompting before any system-level install:

- **uv** (manages Python + virtualenv + deps) — auto-installed if missing
- **Ollama** — auto-installed via `brew` (macOS), the official installer (Linux, needs sudo), or `winget` (Windows, needs UAC)
- The default model (`llama3.1:8b`, ~5 GB) — pulled if missing
- Python dependencies — installed via `uv sync`

After setup, `uv run philosopher_agents.py` is the only command you need to launch a discussion.

## Requirements

- **macOS**: [Homebrew](https://brew.sh) for auto-install of Ollama (otherwise the script prints a download link)
- **Linux**: sudo access for the official Ollama installer
- **Windows**: `winget` (built into Windows 10 1809+ and Windows 11)
- An internet connection for the initial setup
- ~6 GB of free disk space (for the model)

## Running without audio

The script defaults to text-to-speech enabled (Edge TTS, falling back to system TTS). To run silently:

```bash
# macOS / Linux
DISABLE_TTS=true uv run philosopher_agents.py
```

```powershell
# Windows PowerShell
$env:DISABLE_TTS='true'; uv run philosopher_agents.py
```

On Linux, `pyttsx3` (the offline TTS fallback) needs `espeak` installed (`sudo apt install espeak`). If you have internet, Edge TTS works out of the box.

## Configuration

Topic, philosophers, round count, and model are currently set inside `main()` in `philosopher_agents.py`. To change them, edit the bottom of that file:

```python
agents: List[Agent] = [
    philosophers["Socrates"],
    philosophers["Nietzsche"],
]
topic = "..."
max_rounds = 10
```

Available philosophers are defined in the `philosophers` dict — Buddha, Descartes, Diogenes, Jesus, Kant, Nietzsche, Lao Tzu, Sartre, Socrates, Carl Jung, Sigmund Freud.

## Output

Each run writes:
- `discussion_results.json` — the full transcript
- `agent_memories.json` — each agent's accumulated memories, private thoughts, and relationship sentiment (loaded on the next run so conversations build on prior ones)

Both are gitignored.

## Troubleshooting

**`Could not connect to Ollama at http://localhost:11434`**
Ollama isn't running. Re-run `./setup.sh` (or `.\setup.ps1`) — it will start the daemon. Or start it manually: `ollama serve` / open the Ollama desktop app.

**`Model 'llama3.1:8b' not found`**
Pull it: `ollama pull llama3.1:8b`. Or pick a model you already have and change `model_name` in `main()`.

**`uv: command not found` after setup.sh ran**
The uv installer adds it to `~/.local/bin`, which may not be on PATH in your current shell. Open a new terminal and try again, or run `export PATH="$HOME/.local/bin:$PATH"`.

**TTS errors / no audio**
Set `DISABLE_TTS=true` to skip audio entirely. Edge TTS needs internet; if it fails the script falls back to `pyttsx3` (system TTS), which needs an audio output device — and on Linux, `espeak` installed.
