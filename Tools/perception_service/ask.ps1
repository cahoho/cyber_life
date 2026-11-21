param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Question,
    [int]$Port = 8710
)

$ErrorActionPreference = "Stop"
$body = @{ question = $Question } | ConvertTo-Json -Compress
$response = Invoke-RestMethod `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body $body `
    -Uri "http://127.0.0.1:$Port/v1/query"

$response | ConvertTo-Json -Depth 8
