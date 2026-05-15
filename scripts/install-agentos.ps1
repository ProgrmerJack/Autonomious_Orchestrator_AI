param(
    [switch]$NoLaunch
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

Push-Location $Root
& $Python -m pip install --upgrade pip
& $Python -m pip install -e ".[dashboard,os-control]"
Pop-Location

Push-Location $Dashboard
if (-not (Test-Path 'node_modules')) {
    npm install
}
npm run build
Pop-Location

$Launcher = Join-Path $Root 'AgentOS.cmd'
Set-Content -Path $Launcher -Encoding ASCII -Value @'
@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-agentos.ps1" -SkipInstall %*
'@

$Desktop = [Environment]::GetFolderPath('Desktop')
if ($Desktop) {
    $ShortcutPath = Join-Path $Desktop 'AgentOS.lnk'
    $Shell = New-Object -ComObject WScript.Shell
    $Shortcut = $Shell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = $Launcher
    $Shortcut.WorkingDirectory = $Root
    $Shortcut.Save()
}

Write-Host "AgentOS installed. Launch with .\AgentOS.cmd."
if (-not $NoLaunch) {
    & $Launcher
}
