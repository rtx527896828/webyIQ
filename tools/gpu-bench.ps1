# Run a benchmark page in a throwaway Chrome instance pinned to one GPU (Chrome on Windows
# binds the whole browser to a single adapter, so the iGPU needs its own instance).
# Waits for the page to POST to /report (tools/serve.mjs must be running), then closes it.
#   powershell -File tools/gpu-bench.ps1 -Gpu AMD -Url "web/index.html?autobench"
# -Extension <dir> also loads an unpacked extension; -Url may then be a chrome-extension:// URL.
# -Browser Edge runs Microsoft Edge instead (it still honors --load-extension; branded Chrome does not).
param(
    [string]$Gpu = "AMD",
    [string]$Url = "web/index.html?autobench",
    [int]$Port = 8765,
    [int]$TimeoutSec = 600,
    [string]$Extension = "",
    [ValidateSet("Chrome", "Edge")][string]$Browser = "Chrome"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$reports = Join-Path $root "models\reports.jsonl"
$profileDir = Join-Path $env:TEMP "webyiq-$($Browser.ToLower())-$Gpu"

$adapter = Get-ChildItem "HKLM:\SOFTWARE\Microsoft\DirectX" | ForEach-Object { Get-ItemProperty $_.PSPath } |
    Where-Object { $_.Description -like "*$Gpu*" } | Select-Object -First 1
if (-not $adapter) { throw "no adapter matching '$Gpu'" }
# Chrome parses --use-adapter-luid as decimal "high,low"
$luid = [int64]$adapter.AdapterLuid
$arg = "--use-adapter-luid={0},{1}" -f ($luid -shr 32), ($luid -band 0xFFFFFFFF)

$exe, $browserPath = if ($Browser -eq "Edge") {
    "msedge.exe", "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
} else {
    "chrome.exe", "C:\Program Files\Google\Chrome\Application\chrome.exe"
}

function Stop-Instance {
    Get-CimInstance Win32_Process -Filter "Name='$exe'" |
        Where-Object { $_.CommandLine -like "*$profileDir*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

Stop-Instance
# an extension is loaded into a fresh profile each run (a reused one may come up without the test tab)
if ($Extension) { Remove-Item -Recurse -Force $profileDir -ErrorAction SilentlyContinue }
$before = if (Test-Path $reports) { (Get-Content $reports).Count } else { 0 }
$sep = if ($Url.Contains("?")) { "&" } else { "?" }
$full = if ($Url -like "chrome-extension://*") { $Url } else { "http://127.0.0.1:$Port/$Url${sep}t=$([DateTimeOffset]::Now.ToUnixTimeSeconds())" }
Write-Host "$($adapter.Description) $arg -> $full"
# developer features: unquantized timestamps
$chromeArgs = @("--user-data-dir=`"$profileDir`"", "--no-first-run", "--no-default-browser-check", $arg, "--enable-webgpu-developer-features")
# no sync: Edge would sign a fresh profile into the Windows account and sync into it
if ($Browser -eq "Edge") { $chromeArgs += "--disable-sync" }
if ($Extension) {
    # branded Chrome ignores --load-extension unless this switch is re-enabled
    $chromeArgs += "--load-extension=`"$((Resolve-Path $Extension).Path)`"", "--disable-features=DisableLoadExtensionCommandLineSwitch"
}
Start-Process $browserPath -ArgumentList ($chromeArgs + $full) | Out-Null

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    if ((Test-Path $reports) -and (Get-Content $reports).Count -gt $before) { break }
}
Stop-Instance
if (-not ((Test-Path $reports) -and (Get-Content $reports).Count -gt $before)) { throw "timed out waiting for report" }
Get-Content $reports -Tail 1
