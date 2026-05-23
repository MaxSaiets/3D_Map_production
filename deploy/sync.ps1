# auto-deploy script: copy changes → git push → server pull → rebuild → verify
# Usage:
#   .\deploy\sync.ps1 "commit message"
#   .\deploy\sync.ps1 "fix: roads"             # just commit + push + deploy
#   .\deploy\sync.ps1 "fix: roads" -SkipBuild  # backend-only change (skip npm build)
#   .\deploy\sync.ps1 -DryRun                  # show what would happen
param(
    [Parameter(Position=0)] [string]$Message = "auto: sync changes",
    [switch]$SkipBuild,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$LOCAL  = "H:\3dMAP_WORK_2.0"
$DEPLOY = "C:\Temp\3dmap_deploy"
$SERVER = "root@209.38.210.197"
$SSH    = @("-o", "StrictHostKeyChecking=no", $SERVER)

function Write-Step($msg) { Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "    FAIL $msg" -ForegroundColor Red }

# ── 1. Detect changed files in local backend + frontend
Write-Step "Detecting changed files"
$backendChanges = git -C "$LOCAL\backend" status --short 2>$null
$frontendChanges = git -C "$LOCAL\frontend" status --short 2>$null
$allChanges = @()
foreach ($line in ($backendChanges -split "`n")) {
    if ($line -match "^\s*[A-Z?]+\s+(.+)$") { $allChanges += "backend/$($matches[1].Trim())" }
}
foreach ($line in ($frontendChanges -split "`n")) {
    if ($line -match "^\s*[A-Z?]+\s+(.+)$") { $allChanges += "frontend/$($matches[1].Trim())" }
}
# Filter out ignored stuff
$allChanges = $allChanges | Where-Object {
    $_ -notmatch "\.git_bak|__pycache__|node_modules|\.next|debug/|temp_|/output/|tsconfig\.tsbuildinfo|analyze_preview"
}

if (-not $allChanges) {
    Write-Host "    Nothing to sync (no changed files in local repo)" -ForegroundColor Yellow
    # Still check git status of deploy repo (maybe pending commit there)
    $deployStatus = git -C $DEPLOY status --short 2>$null
    if (-not $deployStatus) {
        Write-Host "    Deploy repo also clean. Done."
        exit 0
    }
}
Write-Ok "Found $($allChanges.Count) changed files"
$allChanges | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }

if ($DryRun) { Write-Host "`n[DryRun] Would copy files, commit, push, deploy. Stopping." -ForegroundColor Yellow; exit 0 }

# ── 2. Copy changed files to deploy repo
Write-Step "Copying files to deploy repo"
foreach ($rel in $allChanges) {
    $src = Join-Path $LOCAL ($rel.Replace("/", "\"))
    $dst = Join-Path $DEPLOY ($rel.Replace("/", "\"))
    if (-not (Test-Path $src)) { continue }
    $dstDir = Split-Path $dst -Parent
    if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
    if (Test-Path $src -PathType Container) {
        Copy-Item $src $dst -Recurse -Force
    } else {
        Copy-Item $src $dst -Force
    }
}
Write-Ok "Copied"

# ── 3. Quick syntax check for any .py files changed
$pyChanges = $allChanges | Where-Object { $_ -match "\.py$" }
if ($pyChanges) {
    Write-Step "Python syntax check"
    foreach ($f in $pyChanges) {
        $abs = Join-Path $LOCAL ($f.Replace("/", "\"))
        $result = python -c "import ast; ast.parse(open(r'$abs', encoding='utf-8').read())" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Syntax error in $f"
            Write-Host $result -ForegroundColor Red
            exit 1
        }
    }
    Write-Ok "All Python files OK"
}

# ── 4. Commit + push from deploy repo
Write-Step "Committing and pushing"
$deployStatus = git -C $DEPLOY status --short 2>&1
if (-not $deployStatus) {
    Write-Host "    Deploy repo clean. Skipping commit." -ForegroundColor Yellow
} else {
    git -C $DEPLOY add -A 2>&1 | Out-Null
    $commitOutput = git -C $DEPLOY commit -m $Message 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Commit failed"; Write-Host $commitOutput -ForegroundColor Red; exit 1
    }
    $pushOutput = git -C $DEPLOY push origin main 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Push failed"; Write-Host $pushOutput -ForegroundColor Red; exit 1
    }
    Write-Ok "Pushed: $($commitOutput -match '\[main [a-f0-9]+\].*' | Out-Null; $matches[0])"
}

# ── 5. Pull on server
Write-Step "Server: git pull"
$pullOut = & ssh @SSH "cd /opt/3dmap && git stash 2>/dev/null; git pull origin main" 2>&1
Write-Host ($pullOut | Out-String).Trim() -ForegroundColor DarkGray
Write-Ok "Server synced"

# ── 6. Frontend rebuild (only if frontend files changed)
$frontendTouched = $allChanges | Where-Object { $_ -match "^frontend/" }
if ($frontendTouched -and -not $SkipBuild) {
    Write-Step "Frontend rebuild"
    $buildOut = & ssh @SSH "cd /opt/3dmap/frontend && rm -rf .next && npm run build 2>&1 | tail -5" 2>&1
    Write-Host ($buildOut | Out-String).Trim() -ForegroundColor DarkGray
    if ($buildOut -match "Failed to compile|error TS") {
        Write-Err "Build FAILED"; exit 1
    }
    Write-Ok "Built"
}

# ── 7. Restart PM2
Write-Step "PM2 restart"
$backendTouched = $allChanges | Where-Object { $_ -match "^backend/" }
$restartTargets = @()
if ($backendTouched) { $restartTargets += "3dmap-backend" }
if ($frontendTouched) { $restartTargets += "3dmap-frontend" }
if (-not $restartTargets) { $restartTargets = @("all") }

foreach ($t in $restartTargets) {
    & ssh @SSH "pm2 restart $t --update-env" 2>&1 | Out-Null
    Write-Ok "Restarted $t"
}

# Wait for services
Start-Sleep -Seconds 10

# ── 8. Verify endpoints
Write-Step "Endpoint verification"
$endpoints = @(
    @{ Name = "Backend API";   Url = "http://127.0.0.1:8000/" },
    @{ Name = "Frontend /";    Url = "http://127.0.0.1:3000/" },
    @{ Name = "Frontend /keychains"; Url = "http://127.0.0.1:3000/keychains" },
    @{ Name = "Public (nginx)";   Url = "http://209.38.210.197/" }
)
$allOk = $true
foreach ($ep in $endpoints) {
    $code = & ssh @SSH "curl -s -o /dev/null -w '%{http_code}' '$($ep.Url)'" 2>&1
    if ($code -eq "200") {
        Write-Ok "$($ep.Name): $code"
    } else {
        Write-Err "$($ep.Name): $code"
        $allOk = $false
    }
}

# ── 9. Final PM2 status
Write-Step "PM2 status"
& ssh @SSH "pm2 status" 2>&1 | Select-String "name|3dmap-" | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }

if ($allOk) {
    Write-Host "`n✓ DEPLOY OK — http://209.38.210.197" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n✗ DEPLOY HAD ISSUES — check logs above" -ForegroundColor Red
    Write-Host "  pm2 logs on server: ssh $SERVER 'pm2 logs --lines 30 --nostream'" -ForegroundColor Yellow
    exit 1
}
