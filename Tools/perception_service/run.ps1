param(
    [ValidateSet("auto", "yolo", "heuristic")]
    [string]$Backend = "auto",
    [int]$Port = 8710,
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$serviceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $serviceRoot

python -m perception_service --backend $Backend --port $Port --device $Device
