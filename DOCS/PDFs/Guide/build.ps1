# Build the VIO lecture notes.
#
#   .\build.ps1             figures + tables + both PDF editions
#   .\build.ps1 -SkipFigures  reuse existing figures/tables (faster)
#
# Outputs:
#   guide.pdf               edition with the worked solutions (Appendix E)
#   guide-nosolutions.pdf   edition without them (AGENTS.md Guardrail 4)
#
# Requirements: MiKTeX (pdflatex on PATH); for figures, the local venv:
#   py -3.14 -m venv .venv
#   .\.venv\Scripts\python.exe -m pip install numpy scipy matplotlib mpmath

param([switch]$SkipFigures)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not $SkipFigures) {
    $py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { throw "Python venv not found at $py (see the header of this script)." }
    & $py scripts\make_figures.py
    if ($LASTEXITCODE -ne 0) { throw 'make_figures.py failed' }
    & $py scripts\make_tables.py
    if ($LASTEXITCODE -ne 0) { throw 'make_tables.py failed' }
    & $py scripts\planar_sims.py      # planar MSCKF and FEJ Monte Carlo (a few minutes)
    if ($LASTEXITCODE -ne 0) { throw 'planar_sims.py failed' }
}

function Invoke-LaTeX([string]$JobName, [string]$Source) {
    for ($pass = 1; $pass -le 3; $pass++) {
        pdflatex -interaction=nonstopmode -halt-on-error "-jobname=$JobName" $Source | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Select-String -Path "$JobName.log" -Pattern '^!' -Context 0, 6 | Select-Object -First 3
            throw "pdflatex failed for $JobName (pass $pass); see $JobName.log"
        }
        # Rebuild the index after passes 1 and 2: page numbers recorded on pass 1 can still move
        # once cross-references resolve, so pass 3 typesets an index from the settled layout.
        if ($pass -le 2) {
            makeindex -q "$JobName.idx" | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "makeindex failed for $JobName" }
        }
    }
    $pages = (Select-String -Path "$JobName.log" -Pattern 'Output written on .*\((\d+) pages').Matches.Groups[1].Value
    Write-Output "$JobName.pdf: $pages pages"
}

Invoke-LaTeX 'guide' 'guide.tex'
Invoke-LaTeX 'guide-nosolutions' '\def\NoSolutions{}\input{guide.tex}'
