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
    # Нормалізуємо trailing slash з git output (директорії часто йдуть як "app/keychains/")
    $relClean = $rel.TrimEnd("/", "\")
    $src = Join-Path $LOCAL ($relClean.Replace("/", "\"))
    $dst = Join-Path $DEPLOY ($relClean.Replace("/", "\"))
    if (-not (Test-Path $src)) { continue }
    if (Test-Path $src -PathType Container) {
        # КРИТИЧНО: для директорій копіюємо ВМІСТ у вже існуючу,
        # а не саму папку (інакше створюється keychains/keychains/)
        if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Path $dst -Force | Out-Null }
        Copy-Item -Path "$src\*" -Destination $dst -Recurse -Force
    } else {
        $dstDir = Split-Path $dst -Parent
        if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
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
    # CRITICAL: STOP the running frontend before building. Building while
    # `next start` is live corrupts .next (missing .next/server/app/*/page.js ->
    # "ENOENT page.js" / "o is not a function" / unstyled 500s). Stop -> rm ->
    # build -> (started after the BUILD_ID gate below).
    $buildOut = & ssh @SSH "pm2 stop 3dmap-frontend >/dev/null 2>&1; cd /opt/3dmap/frontend && rm -rf .next && node_modules/.bin/next build > /tmp/3dmap_build.log 2>&1; echo EXIT=`$?; tail -6 /tmp/3dmap_build.log" 2>&1
    Write-Host ($buildOut | Out-String).Trim() -ForegroundColor DarkGray
    $buildId = (& ssh @SSH "test -f /opt/3dmap/frontend/.next/BUILD_ID && echo OK || echo MISSING" 2>&1 | Out-String).Trim()
    if ($buildOut -match "EXIT=[^0]" -or $buildId -notmatch "OK") {
        Write-Err "Build FAILED (.next/BUILD_ID=$buildId) — frontend left stopped. Full log: ssh $SERVER 'cat /tmp/3dmap_build.log'"
        exit 1
    }
    # Build OK -> bring the frontend back up onto the fresh .next.
    & ssh @SSH "pm2 start 3dmap-frontend >/dev/null 2>&1; pm2 restart 3dmap-frontend --update-env >/dev/null 2>&1" 2>&1 | Out-Null
    Write-Ok "Built (BUILD_ID present) + frontend started"
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

# ── 7b. Frontend self-heal: `next start` sometimes loses a compiled module on a
# restart race right after a build (Cannot find module next/dist/compiled/cookie),
# serving 500s. Re-check and restart once if needed.
if ($frontendTouched) {
    for ($i = 1; $i -le 2; $i++) {
        $fe = & ssh @SSH "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3000/" 2>&1
        if ($fe -eq "200") { break }
        Write-Host "    Frontend returned $fe — restarting (attempt $i)" -ForegroundColor Yellow
        & ssh @SSH "pm2 restart 3dmap-frontend --update-env" 2>&1 | Out-Null
        Start-Sleep -Seconds 8
    }
}

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
