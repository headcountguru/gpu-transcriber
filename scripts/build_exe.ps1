# Build the Windows one-folder app and check it starts.
#   powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
# Works from any directory. Needs the project's .venv with the "app" and "build" extras installed:
#   .venv\Scripts\python.exe -m pip install -e ".[app,build]"
# The script's exit code is 0 on success, otherwise the failing step's own exit code
# (PyInstaller's, or the built app's --selftest), or 1 for problems found by this script.
$ErrorActionPreference = "Stop"

# Not Write-Error: with $ErrorActionPreference = "Stop" it throws, which would turn every
# failure into exit code 1 and lose the real code.
function Stop-Build([string]$Message, [int]$Code) {
    [Console]::Error.WriteLine($Message)
    exit $Code
}

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$spec = Join-Path $root "packaging\gpu-transcriber.spec"
$exe = Join-Path $root "dist\gpu-transcriber\gpu-transcriber.exe"

if (-not (Test-Path $python)) {
    Stop-Build "The virtual environment is missing: $python was not found. Create it first (python -m venv .venv) and install the project's requirements." 1
}

Set-Location $root

foreach ($folder in @("build", "dist")) {
    $path = Join-Path $root $folder
    if (Test-Path $path) {
        Remove-Item -Recurse -Force $path
    }
}

& $python -m PyInstaller --noconfirm --workpath (Join-Path $root "build") --distpath (Join-Path $root "dist") $spec
if ($LASTEXITCODE -ne 0) {
    Stop-Build "PyInstaller failed with exit code $LASTEXITCODE." $LASTEXITCODE
}

if (-not (Test-Path $exe)) {
    Stop-Build "The build finished but $exe was not created." 1
}

# The exe is a windowed program, so start it and wait for it explicitly to get its exit code.
Write-Host "Running the self-test of the built app..."
$test = Start-Process -FilePath $exe -ArgumentList "--selftest" -Wait -PassThru -NoNewWindow
if ($test.ExitCode -ne 0) {
    Stop-Build "The built app's self-test failed with exit code $($test.ExitCode)." $test.ExitCode
}
Write-Host "Self-test passed (exit code 0)."

$bytes = (Get-ChildItem -Recurse -File (Split-Path -Parent $exe) | Measure-Object -Property Length -Sum).Sum
Write-Host ("Built: {0}" -f $exe)
Write-Host ("Folder size: {0:N1} MB" -f ($bytes / 1MB))
