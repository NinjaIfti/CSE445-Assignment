<#
.SYNOPSIS
    Windows-side prerequisite for CSE445 Assignment 3: installs WSL2 + Ubuntu.

.DESCRIPTION
    Everything else in this project runs INSIDE WSL. This script only handles the part
    that must happen on the Windows side and needs Administrator rights.

    Run it from an elevated PowerShell (right-click PowerShell -> Run as Administrator):
        powershell -ExecutionPolicy Bypass -File setup_windows.ps1

    A reboot is normally required after the first install. After rebooting, open Ubuntu,
    cd to this repository and run:
        bash setup_wsl.sh
#>

[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu-22.04"
)

$ErrorActionPreference = "Stop"

function Write-Step { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok   { param($m) Write-Host "[ok] $m"   -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "[warn] $m" -ForegroundColor Yellow }

# --- Administrator check ---------------------------------------------------------------
$identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Warn "This script must run as Administrator. Re-launch PowerShell with 'Run as Administrator'."
    exit 1
}

Write-Step "Checking for an existing WSL installation"
$wslInstalled = $true
try {
    $status = & wsl.exe --status 2>&1 | Out-String
    if ($status -match "not installed") { $wslInstalled = $false }
} catch {
    $wslInstalled = $false
}

if (-not $wslInstalled) {
    Write-Step "Installing WSL2 with $Distro (this downloads several hundred MB)"
    & wsl.exe --install -d $Distro
    Write-Warn "A REBOOT is required to finish the WSL installation."
    Write-Host  "After rebooting, launch Ubuntu once to create your Linux user, then run:"
    Write-Host  "    cd /mnt/<drive>/path/to/this/repo && bash setup_wsl.sh" -ForegroundColor White
    exit 0
}

Write-Ok "WSL is already installed"
& wsl.exe --set-default-version 2 | Out-Null

Write-Step "Installed distributions"
& wsl.exe -l -v

$distros = (& wsl.exe -l -q) -join "`n"
if ($distros -notmatch [Regex]::Escape($Distro.Split('-')[0])) {
    Write-Step "Installing distribution $Distro"
    & wsl.exe --install -d $Distro
} else {
    Write-Ok "$Distro (or another Ubuntu) is present"
}

Write-Step "Checking GPU passthrough"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    Write-Ok "NVIDIA driver present. WSL2 exposes it at /usr/lib/wsl/lib; setup_wsl.sh will pick a CUDA wheel."
} else {
    Write-Warn "No NVIDIA driver found. setup_wsl.sh will install the CPU PyTorch build."
}

Write-Host @"

======================================================================
Windows-side setup complete.

Next:
  1. Open the Ubuntu terminal.
  2. cd to this repository, e.g.
       cd /mnt/k/'CSE445 Assignment'
  3. bash setup_wsl.sh
======================================================================
"@ -ForegroundColor Green
