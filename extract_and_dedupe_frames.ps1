# Extract MP4 frames, then deduplicate nearly identical consecutive frames.
#
# Env knobs:
#   $env:PYTHON              # python interpreter (else venv at <root>\.venv_extract or PATH python)
#   $env:EXTRACT_FRAME_OUT   # frame output dir (else --out arg, else <here>\data\images)
#   $env:DEDUP_OUT           # dedupe output dir (default <here>\data\images_deduped)

$ErrorActionPreference = 'Stop'

$Here = $PSScriptRoot
$Root = Split-Path -Parent $Here

if (-not $env:PYTHON) {
    $venvPy = Join-Path $Root '.venv_extract\Scripts\python.exe'
    if (Test-Path $venvPy) {
        $Python = $venvPy
    } else {
        $py = Get-Command python -ErrorAction SilentlyContinue
        if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
        if (-not $py) { throw "could not locate python on PATH; set `$env:PYTHON" }
        $Python = $py.Source
    }
} else {
    $Python = $env:PYTHON
}

$defaultFrameOut = Join-Path $Here 'data\images'
$frameOut = $env:EXTRACT_FRAME_OUT

if (-not $frameOut) {
    for ($i = 0; $i -lt $args.Count; $i++) {
        $a = $args[$i]
        if ($a -eq '--out' -and ($i + 1) -lt $args.Count) {
            $frameOut = $args[$i + 1]
            break
        }
        if ($a -like '--out=*') {
            $frameOut = $a.Substring(6)
            break
        }
    }
}
if (-not $frameOut) { $frameOut = $defaultFrameOut }

if ($env:DEDUP_OUT) {
    $dedupOut = $env:DEDUP_OUT
} else {
    $dedupOut = Join-Path $Here 'data\images_deduped'
}

Write-Host "==> extract -> $frameOut"
& $Python (Join-Path $Here 'extract_mp4_frames.py') @args
if ($LASTEXITCODE -ne 0) { throw "extract_mp4_frames.py failed" }

Write-Host ""
Write-Host "==> dedupe  (from $frameOut) -> $dedupOut"
& $Python (Join-Path $Here 'dedupe_similar_frames.py') --in $frameOut --out $dedupOut
if ($LASTEXITCODE -ne 0) { throw "dedupe_similar_frames.py failed" }
