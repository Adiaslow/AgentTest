# Bootstrap for Windows: installs uv if missing, then runs setup.py.
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "-> Installing uv (https://docs.astral.sh/uv/)..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # uv installs to %USERPROFILE%\.local\bin by default; make it available now
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "uv install succeeded but 'uv' is not on PATH. Open a new PowerShell window and re-run .\setup.ps1"
        exit 1
    }
}

uv run --no-project setup.py
exit $LASTEXITCODE
