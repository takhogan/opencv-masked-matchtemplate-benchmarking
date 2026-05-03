# Build cv2 from original_opencv and/or new_opencv and print the resulting
# cv2 package paths. Does not run any benchmarks — use .\run.ps1 for that.
#
# Usage:
#   .\build_opencvs.ps1                    # build both (original + new)
#   .\build_opencvs.ps1 original           # build only original
#   .\build_opencvs.ps1 new                # build only new
#   .\build_opencvs.ps1 --fresh            # wipe both build dirs first
#   .\build_opencvs.ps1 --clean-new        # wipe only new build, then rebuild new
#
# Env knobs: $env:PYTHON, $env:JOBS, $env:WITH_CUDA (ON/OFF), $env:CMAKE_EXTRA

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot

if (-not $env:PYTHON) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
    if (-not $py) { throw "could not locate python on PATH; set `$env:PYTHON" }
    $Python = $py.Source
} else {
    $Python = $env:PYTHON
}

if (-not $env:JOBS) {
    $Jobs = [Environment]::ProcessorCount
} else {
    $Jobs = [int]$env:JOBS
}

if (-not $env:WITH_CUDA) {
    if (Get-Command nvcc -ErrorAction SilentlyContinue) { $WithCuda = 'ON' } else { $WithCuda = 'OFF' }
} else {
    $WithCuda = $env:WITH_CUDA
}

$Targets = @()
$Fresh = $false
$CleanNew = $false
$TargetsExplicit = $false
foreach ($arg in $args) {
    switch ($arg) {
        'original'    { $Targets += 'original'; $TargetsExplicit = $true }
        'new'         { $Targets += 'new';      $TargetsExplicit = $true }
        '--fresh'     { $Fresh = $true }
        '--clean-new' { $CleanNew = $true }
        '-h'          { Get-Content -Path $PSCommandPath -TotalCount 13 | Select-Object -Skip 1; exit 0 }
        '--help'      { Get-Content -Path $PSCommandPath -TotalCount 13 | Select-Object -Skip 1; exit 0 }
        default       { Write-Error "unknown arg: $arg"; exit 2 }
    }
}
if ($CleanNew -and -not $TargetsExplicit) { $Targets = @('new') }
if ($Targets.Count -eq 0) { $Targets = @('original', 'new') }

Write-Host "==> python  : $Python"
Write-Host "==> jobs    : $Jobs"
Write-Host "==> CUDA    : $WithCuda"
Write-Host "==> targets : $($Targets -join ' ')"
Write-Host ""

$PyInc    = & $Python -c "import sysconfig; print(sysconfig.get_path('include'))"
$PyLib    = & $Python -c "import sysconfig,os; libdir=sysconfig.get_config_var('LIBDIR') or sysconfig.get_config_var('installed_base'); ldlib=sysconfig.get_config_var('LDLIBRARY') or ''; print(os.path.join(libdir, ldlib) if ldlib else '')"
$NumpyInc = & $Python -c "import numpy; print(numpy.get_include())"

function Build-One {
    param([string]$Tag)

    switch ($Tag) {
        'original' { $tree = Join-Path $Root 'original_opencv' }
        'new'      { $tree = Join-Path $Root 'new_opencv' }
        default    { throw "bad tag $Tag" }
    }

    $src     = Join-Path $tree 'opencv'
    $contrib = Join-Path $tree 'opencv_contrib\modules'
    $build   = Join-Path $src 'build'

    if (-not (Test-Path $src))     { Write-Warning "missing $src — skipping $Tag";     return }
    if (-not (Test-Path $contrib)) { Write-Warning "missing $contrib — skipping $Tag"; return }

    $wipe = $false
    if ($Fresh) { $wipe = $true }
    if ($CleanNew -and $Tag -eq 'new') { $wipe = $true }
    if ($wipe -and (Test-Path $build)) {
        Write-Host "==> [$Tag] clean: removing $build"
        Remove-Item -Recurse -Force $build
    }

    Write-Host "==> [$Tag] configuring in $build"
    if (-not (Test-Path $build)) { New-Item -ItemType Directory -Path $build | Out-Null }

    Push-Location $build
    try {
        $cmakeArgs = @(
            $src,
            '-DCMAKE_BUILD_TYPE=Release',
            "-DOPENCV_EXTRA_MODULES_PATH=$contrib",
            '-DWITH_OPENCL=ON',
            "-DWITH_CUDA=$WithCuda",
            '-DBUILD_opencv_python3=ON',
            '-DBUILD_opencv_python_bindings_generator=ON',
            "-DPYTHON3_EXECUTABLE=$Python",
            "-DPYTHON3_INCLUDE_DIR=$PyInc",
            "-DPYTHON3_NUMPY_INCLUDE_DIRS=$NumpyInc",
            '-DBUILD_TESTS=OFF',
            '-DBUILD_PERF_TESTS=OFF',
            '-DBUILD_EXAMPLES=OFF',
            '-DBUILD_DOCS=OFF',
            '-DBUILD_JAVA=OFF',
            '-DBUILD_opencv_apps=OFF',
            '-DCMAKE_C_FLAGS=/w',
            '-DCMAKE_CXX_FLAGS=/w'
        )
        if ($PyLib) { $cmakeArgs += "-DPYTHON3_LIBRARY=$PyLib" }
        if ($env:CMAKE_EXTRA) { $cmakeArgs += $env:CMAKE_EXTRA.Split(' ') }

        & cmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed for $Tag" }

        Write-Host "==> [$Tag] building (-j$Jobs)"
        & cmake --build . --config Release -j $Jobs --target opencv_python3
        if ($LASTEXITCODE -ne 0) { throw "cmake build failed for $Tag" }
    } finally {
        Pop-Location
    }
}

function Find-Cv2Path {
    param([string]$Build)
    $loader = Join-Path $Build 'python_loader'
    if (Test-Path $loader) { return $loader }
    $hit = Get-ChildItem -Path $Build -Recurse -ErrorAction SilentlyContinue `
        -Include 'cv2*.pyd','cv2*.so','cv2*.dylib' | Select-Object -First 1
    if ($hit) { return $hit.DirectoryName }
    return $null
}

foreach ($tag in $Targets) { Build-One $tag }

Write-Host ""
Write-Host "==> locating cv2 packages"
foreach ($tag in $Targets) {
    switch ($tag) {
        'original' { $localBuild = Join-Path $Root 'original_opencv\opencv\build' }
        'new'      { $localBuild = Join-Path $Root 'new_opencv\opencv\build' }
    }
    $path = Find-Cv2Path $localBuild
    if ($path) {
        Write-Host "    $tag -> $path"
        $env:PYTHONPATH = $path
        & $Python -c "import cv2; print('       ok: cv2', cv2.__version__, 'from', cv2.__file__)"
        if ($LASTEXITCODE -ne 0) { Write-Host "       !! import check failed" }
        Remove-Item Env:PYTHONPATH
    } else {
        Write-Host "    $tag -> (not found under $localBuild)"
    }
}

Write-Host ""
Write-Host "==> done. Run .\run.ps1 to execute the benchmark."
