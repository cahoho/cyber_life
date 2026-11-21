param(
    [string]$Model = "yolov8n.pt"
)

$ErrorActionPreference = "Stop"
$serviceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelRoot = Join-Path $serviceRoot "models"
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null

Push-Location $modelRoot
try {
    python -c "from ultralytics import YOLO; import sys; YOLO(sys.argv[1])" $Model
}
finally {
    Pop-Location
}

$modelPath = Join-Path $modelRoot $Model
if (-not (Test-Path $modelPath)) {
    throw "Model download did not produce $modelPath"
}

Get-Item $modelPath | Select-Object FullName, Length
