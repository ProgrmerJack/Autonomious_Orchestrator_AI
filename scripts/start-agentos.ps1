param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 5173,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Dashboard = Join-Path $Root 'apps\dashboard'

if (-not (Test-Path $Python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.11 -m venv (Join-Path $Root '.venv')
    }
    else {
        python -m venv (Join-Path $Root '.venv')
    }
}

if (-not $SkipInstall) {
    Push-Location $Root
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -e ".[dashboard]"
    Pop-Location
    Push-Location $Dashboard
    if (-not (Test-Path 'node_modules')) {
        npm install
    }
    Pop-Location
}

Push-Location $Root
& $Python -m agentos_orchestrator launch `
    --api-port $ApiPort `
    --ui-port $UiPort `
    --skip-npm-install
Pop-Location