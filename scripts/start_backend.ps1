# 重启 backend（杀掉旧进程 → 启动 uvicorn → 等待健康检查）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\start_backend.ps1

$ErrorActionPreference = "SilentlyContinue"

# 1. 只杀掉 backend 相关 python 进程（不影响其他 python 任务）
$old = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'uvicorn' -and $_.CommandLine -match 'backend\.main' }
foreach ($p in $old) { Stop-Process -Id $p.ProcessId -Force }
if ($old) { Write-Host ("已停止旧 backend 进程 x{0}" -f $old.Count) }

Start-Sleep -Seconds 1

# 2. 启动新 backend（独立进程，日志写入临时目录）
$env:HF_HUB_OFFLINE = "1"
Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","backend.main:app","--port","8000" `
    -WorkingDirectory "D:\Projects\Discussion_Idea\mini_smolagents" `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$env:TEMPackend_out.log" `
    -RedirectStandardError "$env:TEMPackend_err.log"

# 3. 轮询健康检查（最多 40 秒）
$t0 = Get-Date
$ok = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 3
        if ($r.status -eq "ok") { $ok = $true; break }
    } catch {}
}

if ($ok) {
    $elapsed = ((Get-Date) - $t0).TotalSeconds
    Write-Host ("backend 启动成功，用时 {0:N1}s，health: ok" -f $elapsed)
} else {
    Write-Host "backend 启动失败，请查看日志："
    Get-Content "$env:TEMPackend_err.log" -Encoding utf8 -Tail 10
}
