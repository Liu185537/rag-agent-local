param(
  [string]$BaseUrl = "http://127.0.0.1:8008",
  [string]$Namespace = "interview_demo"
)

$ErrorActionPreference = "Stop"

Write-Host "正在执行就绪检查..."
python scripts/readiness_check.py --base-url $BaseUrl --namespace $Namespace

Write-Host "正在执行面试演示流程..."
python scripts/interview_demo.py --base-url $BaseUrl --namespace $Namespace

Write-Host "执行完成。请打开 Dashboard：$BaseUrl/dashboard"
