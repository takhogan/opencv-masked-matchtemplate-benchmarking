# Recreate the workspace on a fresh machine (Windows / PowerShell).
#
# Layout produced (relative to the parent of this script's directory):
#   <root>/
#     opencv-masked-matchtemplate-benchmarking/   # this repo (already cloned by you)
#     new_opencv/
#       opencv/              # takhogan/opencv  @ feature/opencl-masked-template-match
#       opencv_contrib/      # takhogan/opencv_contrib @ cuda-masked-template-match
#     original_opencv/
#       opencv/              # opencv/opencv @ 4.x
#       opencv_contrib/      # opencv/opencv_contrib @ 4.x
#
# Usage:
#   .\setup.ps1                          # https clones (default)
#   $env:USE_SSH=1; .\setup.ps1          # use git@github.com for the takhogan forks

$ErrorActionPreference = 'Stop'

$Here = $PSScriptRoot
$Root = Split-Path -Parent $Here

if ($env:USE_SSH -eq '1') {
    $ForkOpencv  = 'git@github.com:takhogan/opencv.git'
    $ForkContrib = 'git@github.com:takhogan/opencv_contrib.git'
} else {
    $ForkOpencv  = 'https://github.com/takhogan/opencv.git'
    $ForkContrib = 'https://github.com/takhogan/opencv_contrib.git'
}

$UpstreamOpencv  = 'https://github.com/opencv/opencv.git'
$UpstreamContrib = 'https://github.com/opencv/opencv_contrib.git'

function Clone-At {
    param([string]$Url, [string]$Dest, [string]$Branch)
    if (Test-Path (Join-Path $Dest '.git')) {
        Write-Host "[skip] $Dest already exists"
        return
    }
    Write-Host "[clone] $Url -> $Dest (branch $Branch)"
    $parent = Split-Path -Parent $Dest
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
    git clone --branch $Branch $Url $Dest
    if ($LASTEXITCODE -ne 0) { throw "git clone failed for $Url" }
}

Clone-At $ForkOpencv      (Join-Path $Root 'new_opencv\opencv')              'feature/opencl-masked-template-match'
Clone-At $ForkContrib     (Join-Path $Root 'new_opencv\opencv_contrib')      'cuda-masked-template-match'
Clone-At $UpstreamOpencv  (Join-Path $Root 'original_opencv\opencv')         '4.x'
Clone-At $UpstreamContrib (Join-Path $Root 'original_opencv\opencv_contrib') '4.x'

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Drop your input videos / frames into $Here\data\"
Write-Host "  2. Build OpenCV:   $Here\build_opencvs.ps1"
Write-Host "  3. Run benchmark:  $Here\run.ps1"
