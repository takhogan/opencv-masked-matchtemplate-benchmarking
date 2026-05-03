# Build cv2 from original_opencv and new_opencv if missing, then run the
# masked-matchTemplate benchmark.
#
#   .\run.ps1              # build if needed, then run
#   .\run.ps1 --fresh      # wipe and rebuild both opencv trees first
#   .\run.ps1 --clean-new  # wipe and rebuild only the new tree first

$ErrorActionPreference = 'Stop'

$Here = $PSScriptRoot
$Root = Split-Path -Parent $Here

if (-not $env:PYTHON) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
    if (-not $py) { throw "could not locate python on PATH; set `$env:PYTHON" }
    $Python = $py.Source
} else {
    $Python = $env:PYTHON
}

$Fresh = $false
$CleanNew = $false
foreach ($arg in $args) {
    switch ($arg) {
        '--fresh'     { $Fresh = $true }
        '--clean-new' { $CleanNew = $true }
        default       { throw "unknown arg: $arg" }
    }
}

function Find-Cv2Path {
    param([string]$Build)
    $loader = Join-Path $Build 'python_loader'
    if (Test-Path $loader) {
        $env:PYTHONPATH = $loader
        & $Python -c 'import cv2' 2>$null
        $ok = ($LASTEXITCODE -eq 0)
        Remove-Item Env:PYTHONPATH
        if ($ok) { return $loader }
    }
    $hit = Get-ChildItem -Path $Build -Recurse -ErrorAction SilentlyContinue `
        -Include 'cv2*.pyd','cv2*.so','cv2*.dylib' | Select-Object -First 1
    if ($hit) {
        $env:PYTHONPATH = $hit.DirectoryName
        & $Python -c 'import cv2' 2>$null
        $ok = ($LASTEXITCODE -eq 0)
        Remove-Item Env:PYTHONPATH
        if ($ok) { return $hit.DirectoryName }
    }
    return $null
}

$OrigBuild = Join-Path $Root 'original_opencv\opencv\build'
$NewBuild  = Join-Path $Root 'new_opencv\opencv\build'

if ($Fresh) {
    $OrigPath = $null
    $NewPath  = $null
} else {
    $OrigPath = Find-Cv2Path $OrigBuild
    $NewPath  = Find-Cv2Path $NewBuild
}
if ($CleanNew) { $NewPath = $null }

$buildTargets = @()
if (-not $OrigPath) { $buildTargets += 'original' }
if (-not $NewPath)  { $buildTargets += 'new' }

if ($buildTargets.Count -gt 0) {
    Write-Host "==> need to build: $($buildTargets -join ' ')"
    $buildArgs = @($buildTargets)
    if ($Fresh)    { $buildArgs += '--fresh' }
    if ($CleanNew) { $buildArgs += '--clean-new' }
    $env:PYTHON = $Python
    & (Join-Path $Here 'build_opencvs.ps1') @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "build_opencvs.ps1 failed" }
    $OrigPath = Find-Cv2Path $OrigBuild
    $NewPath  = Find-Cv2Path $NewBuild
}

if (-not $OrigPath -or -not $NewPath) {
    Write-Error "still missing a cv2 build (orig='$OrigPath' new='$NewPath')"
    exit 1
}

Write-Host "==> original cv2: $OrigPath"
Write-Host "==> new      cv2: $NewPath"
Write-Host ""

& $Python (Join-Path $Here 'masked_template_match.py') `
    --original-cv2 $OrigPath `
    --new-cv2      $NewPath
exit $LASTEXITCODE
