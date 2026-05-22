"""Cross-platform bootstrap: installs Ollama if missing, pulls the model, syncs deps.

Invoked by setup.sh / setup.ps1 after uv is installed. Uses stdlib only so it
can run before `uv sync` provisions project deps.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_STARTUP_TIMEOUT = 30  # seconds


def _style(code: str, msg: str) -> str:
    if sys.stdout.isatty() and platform.system() != "Windows":
        return f"\033[{code}m{msg}\033[0m"
    return msg


def ok(msg: str) -> None:
    print(f"{_style('32', '✓')} {msg}")


def info(msg: str) -> None:
    print(f"{_style('36', '→')} {msg}")


def warn(msg: str) -> None:
    print(f"{_style('33', '⚠')} {msg}")


def err(msg: str) -> None:
    print(f"{_style('31', '✗')} {msg}", file=sys.stderr)


def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    if not sys.stdin.isatty():
        return default
    while True:
        try:
            ans = input(f"{question} {suffix} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def find_ollama() -> Optional[str]:
    """Return path to ollama binary, checking PATH and known install locations."""
    found = shutil.which("ollama")
    if found:
        return found
    system = platform.system()
    candidates = []
    if system == "Windows":
        candidates.append(
            os.path.expandvars(r"%LocalAppData%\Programs\Ollama\ollama.exe")
        )
    elif system == "Darwin":
        candidates.extend(["/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"])
    elif system == "Linux":
        candidates.extend(["/usr/local/bin/ollama", "/usr/bin/ollama"])
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def install_ollama() -> bool:
    """Attempt to install Ollama using the OS-appropriate method. Returns True on success."""
    system = platform.system()

    if system == "Darwin":
        if shutil.which("brew"):
            cmd_desc = "brew install ollama"
            if not prompt_yes_no(f"Install Ollama via `{cmd_desc}`?"):
                return False
            info(f"Running: {cmd_desc}")
            rc = subprocess.call(["brew", "install", "ollama"])
            return rc == 0
        warn("Homebrew not found — cannot auto-install on macOS.")
        print("Download the installer from: https://ollama.com/download")
        return False

    if system == "Linux":
        cmd_desc = "curl -fsSL https://ollama.com/install.sh | sh"
        print("The official Linux installer will run with sudo to install to /usr/local/bin")
        print("and set up a systemd service.")
        if not prompt_yes_no(f"Install Ollama via `{cmd_desc}`?"):
            return False
        info(f"Running: {cmd_desc}")
        rc = subprocess.call(cmd_desc, shell=True)
        return rc == 0

    if system == "Windows":
        if shutil.which("winget"):
            cmd_desc = "winget install Ollama.Ollama"
            print("Windows will prompt for installer permissions (UAC).")
            if not prompt_yes_no(f"Install Ollama via `{cmd_desc}`?"):
                return False
            info(f"Running: {cmd_desc}")
            rc = subprocess.call(
                [
                    "winget",
                    "install",
                    "--id",
                    "Ollama.Ollama",
                    "-e",
                    "--accept-source-agreements",
                    "--accept-package-agreements",
                ]
            )
            return rc == 0
        warn("winget not found — cannot auto-install on Windows.")
        print("Download the installer from: https://ollama.com/download")
        return False

    warn(f"Unsupported OS: {system}")
    print("Install Ollama from: https://ollama.com/download")
    return False


def is_ollama_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def ollama_has_model(model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
        return any(m.get("name", "").startswith(model) for m in data.get("models", []))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


def start_ollama(ollama_bin: str) -> bool:
    info("Starting Ollama daemon in the background...")
    try:
        if platform.system() == "Windows":
            subprocess.Popen(
                [ollama_bin, "serve"],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    except FileNotFoundError:
        return False

    for _ in range(OLLAMA_STARTUP_TIMEOUT):
        if is_ollama_running():
            return True
        time.sleep(1)
    return False


def main() -> int:
    print("Philosopher Agents — setup\n")

    ollama_bin = find_ollama()
    if ollama_bin is None:
        warn("Ollama is not installed.")
        if not install_ollama():
            err("Ollama install was skipped or failed. Re-run this setup once Ollama is installed.")
            return 1
        ollama_bin = find_ollama()
        if ollama_bin is None:
            err("Ollama install completed but the binary still isn't on PATH.")
            print("Open a new terminal and re-run setup, or add Ollama's install dir to PATH manually.")
            return 1
    ok(f"Ollama installed ({ollama_bin})")

    if not is_ollama_running():
        if not start_ollama(ollama_bin):
            err(
                f"Ollama is installed but did not respond within {OLLAMA_STARTUP_TIMEOUT}s.\n"
                f"Try starting it manually: {ollama_bin} serve"
            )
            return 1
    ok(f"Ollama running at {OLLAMA_URL}")

    if not ollama_has_model(MODEL):
        info(f"Pulling {MODEL} (~5 GB — this can take a few minutes)...")
        rc = subprocess.call([ollama_bin, "pull", MODEL])
        if rc != 0:
            err(f"Failed to pull {MODEL}")
            return rc
    ok(f"Model {MODEL} available")

    info("Installing Python dependencies (uv sync)...")
    rc = subprocess.call(["uv", "sync"])
    if rc != 0:
        err("uv sync failed")
        return rc
    ok("Python dependencies installed")

    print()
    ok("Setup complete!")
    print("\nRun the discussion with:")
    print("  uv run philosopher_agents.py")
    print("\nOr without audio:")
    if platform.system() == "Windows":
        print("  $env:DISABLE_TTS='true'; uv run philosopher_agents.py")
    else:
        print("  DISABLE_TTS=true uv run philosopher_agents.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
