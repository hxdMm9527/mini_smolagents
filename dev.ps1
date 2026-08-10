# mini_smolagents 一键开发启动：backend (Docker, 热重载) + frontend (Vite, HMR)
# 用法:  powershell -ExecutionPolicy Bypass -File dev.ps1
# 之后所有代码改动自动热更新，无需重启。

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "==> [1/3] 检查 Docker..."
if (-not (docker info *> $null)) {
    Write-Host "[ERROR] Docker 未运行，请先启动 Docker Desktop" -ForegroundColor Red
    exit 1
}

Write-Host "==> [2/3] 启动 backend (Docker, uvicorn --reload)..."
docker compose -f "$root\docker-compose.yml" up -d --build backend
if (-not $?) { exit 1 }
Write-Host "    backend: http://127.0.0.1:8000"

Write-Host "==> [3/3] 启动 frontend (Vite)..."
Write-Host "    frontend: http://127.0.0.1:5173"
Write-Host ""
Write-Host "提示: 代码改动自动热更新。backend 日志: docker logs -f ms-backend"
Push-Location "$root\frontend"
try {
    npm run dev
}
finally {
    Pop-Location
}
