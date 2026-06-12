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
# КРИТИЧНО: шляхи типу app\[locale]\keychains містять дужки, які PowerShell
# трактує як wildcard. Тому ВСЮДИ -LiteralPath, а для тек — robocopy (він не
# робить glob і коректно працює з [ ]). Інакше бракетні файли мовчки НЕ копіюються.
Write-Step "Copying files to deploy repo"
$copyErrors = 0
foreach ($rel in $allChanges) {
    # git показує перейменування як "old -> new" — беремо НОВИЙ шлях.
    $relPath = $rel
    if ($relPath -match " -> ") {
        $relPath = ($relPath -split " -> ")[-1].Trim()
        # відновлюємо префікс репо (frontend/ чи backend/) якщо git його прибрав
        if ($rel -match "^(frontend|backend)/" -and $relPath -notmatch "^(frontend|backend)/") {
            $relPath = "$($matches[1])/$relPath"
        }
    }
    # Нормалізуємо trailing slash з git output (директорії часто йдуть як "app/keychains/")
    $relClean = $relPath.TrimEnd("/", "\")
    $src = Join-Path $LOCAL ($relClean.Replace("/", "\"))
    $dst = Join-Path $DEPLOY ($relClean.Replace("/", "\"))
    if (-not (Test-Path -LiteralPath $src)) { continue }
    if (Test-Path -LiteralPath $src -PathType Container) {
        # Копіюємо ВМІСТ теки у відповідну теку deploy (robocopy /E = з підтеками).
        # .NET CreateDirectory — bracket-safe (New-Item -Path робить glob).
        [System.IO.Directory]::CreateDirectory($dst) | Out-Null
        robocopy $src $dst /E /NFL /NDL /NJH /NJS /NP /XD node_modules .next __pycache__ /XF "*.pyc" | Out-Null
        if ($LASTEXITCODE -ge 8) { Write-Err "robocopy failed for $relClean (code $LASTEXITCODE)"; $copyErrors++ }
        $global:LASTEXITCODE = 0
    } else {
        $dstDir = [System.IO.Path]::GetDirectoryName($dst)
        [System.IO.Directory]::CreateDirectory($dstDir) | Out-Null
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}
if ($copyErrors -gt 0) { Write-Err "$copyErrors copy error(s) — aborting"; exit 1 }
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
    $buildOut = & ssh @SSH "pm2 stop 3dmap-frontend >/dev/null 2>&1; pkill -9 -f 'next-server' 2>/dev/null; pkill -9 -f 'next start' 2>/dev/null; sleep 2; cd /opt/3dmap/frontend && npm install --no-audit --no-fund > /tmp/3dmap_npm.log 2>&1; echo NPM=`$?; rm -rf .next node_modules/.cache && node_modules/next/dist/bin/next build > /tmp/3dmap_build.log 2>&1; echo EXIT=`$?; tail -6 /tmp/3dmap_build.log" 2>&1
    Write-Host ($buildOut | Out-String).Trim() -ForegroundColor DarkGray
    $buildId = (& ssh @SSH "test -f /opt/3dmap/frontend/.next/BUILD_ID && echo OK || echo MISSING" 2>&1 | Out-String).Trim()
    # Авто-відновлення: пошкоджений node_modules (бракує next/dist/... модулів)
    # валить білд на etапі type-check. `npm install` цього не лікує — потрібен
    # повний `npm ci`. Робимо один раз і перебілдимо.
    if (($buildOut -match "EXIT=[^0]" -or $buildId -notmatch "OK") -and
        ($buildOut -match "Cannot find module|MODULE_NOT_FOUND|require stack")) {
        Write-Host "    Corrupted node_modules detected — npm ci + rebuild..." -ForegroundColor Yellow
        $buildOut = & ssh @SSH "cd /opt/3dmap/frontend && npm ci > /tmp/3dmap_npm.log 2>&1; echo NPM=`$?; rm -rf .next node_modules/.cache && node_modules/next/dist/bin/next build > /tmp/3dmap_build.log 2>&1; echo EXIT=`$?; tail -6 /tmp/3dmap_build.log" 2>&1
        Write-Host ($buildOut | Out-String).Trim() -ForegroundColor DarkGray
        $buildId = (& ssh @SSH "test -f /opt/3dmap/frontend/.next/BUILD_ID && echo OK || echo MISSING" 2>&1 | Out-String).Trim()
    }
    if ($buildOut -match "EXIT=[^0]" -or $buildId -notmatch "OK") {
        Write-Err "Build FAILED (.next/BUILD_ID=$buildId) — frontend left stopped. Full log: ssh $SERVER 'cat /tmp/3dmap_build.log'"
        exit 1
    }
    # Build OK -> bring the frontend back up onto the fresh .next, then WARM UP:
    # the first request after a cold start intermittently hits Next's lazy
    # `next/dist/compiled/cookie` require race (-> a one-off 500). We consume
    # that ourselves by curling origin until it returns 200, so real users
    # (via Cloudflare) never see it.
    # ВАЖЛИВО: одинарні лапки PS → bash отримує $(seq) і $c буквально (інакше
    # PowerShell сам обчислює $(seq 1 20) і падає "seq not recognized").
    & ssh @SSH 'pm2 start 3dmap-frontend >/dev/null 2>&1; pm2 restart 3dmap-frontend --update-env >/dev/null 2>&1; for i in $(seq 1 25); do c=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3000/); [ "$c" = "200" ] && break; sleep 1; done; echo warmup=$c' 2>&1 | Out-Null
    Write-Ok "Built + started + warmed up"
}

# ── 7. Restart PM2
Write-Step "PM2 restart"
$backendTouched = $allChanges | Where-Object { $_ -match "^backend/" }
$restartTargets = @()
if ($backendTouched) { $restartTargets += "3dmap-backend" }
# ВАЖЛИВО: фронт НЕ рестартуємо, якщо крок 6 щойно збілдив+запустив+прогрів його.
# Зайвий restart одразу після старту запускав cookie-race 500 → 7b рестартував
# падаючий процес → pm2 у рестарт-циклі → guard-ребілд ловив ENOTEMPTY від
# недобитого next start, що писав у .next (двічі поклало сайт 2026-06-12).
if ($frontendTouched -and $SkipBuild) { $restartTargets += "3dmap-frontend" }
if (-not $restartTargets -and -not $frontendTouched) { $restartTargets = @("all") }
if ($frontendTouched -and -not $SkipBuild) { Write-Ok "Frontend freshly built+started in step 6 — skipping redundant restart" }

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

# ── 7c. Build-integrity guard (stale/partial-chunk → 400 → карта+вхід падають).
# Перевіряємо, що webpack-чанк, на який посилається ВІДДАНИЙ HTML, реально є на
# диску і повертає 200. Якщо ні — один чистий ребілд (rm .next + build + restart).
if ($frontendTouched) {
    Write-Step "Build integrity (stale-chunk guard)"
    # WAIT for the frontend to actually be up (pm2 restart may still be settling),
    # THEN compare the served HTML's webpack ref against the on-disk file (file
    # existence, not HTTP — avoids false MISMATCH on a transient 500/000 mid-restart).
    $intCmd = 'cd /opt/3dmap/frontend; for i in $(seq 1 20); do up=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3000/keychains); [ "$up" = "200" ] && break; sleep 1; done; html=$(curl -s http://127.0.0.1:3000/keychains | grep -oE "webpack-[a-z0-9]+\.js" | head -1); if [ -z "$html" ]; then echo "MISMATCH up=$up html=empty"; elif [ -f ".next/static/chunks/$html" ]; then echo "MATCH $html"; else echo "MISMATCH html=$html not-on-disk"; fi'
    $check = (& ssh @SSH $intCmd 2>&1 | Out-String).Trim()
    Write-Host "    $check" -ForegroundColor DarkGray
    if ($check -notmatch '^MATCH') {
        Write-Host "    Stale/partial build — clean rebuild (with npm ci fallback)..." -ForegroundColor Yellow
        # Clean rebuild; if the build dies on a corrupted node_modules
        # (MODULE_NOT_FOUND / 'No such file'), restore with npm ci and rebuild.
        $rb = & ssh @SSH 'cd /opt/3dmap/frontend && pm2 stop 3dmap-frontend >/dev/null 2>&1; pkill -9 -f "next-server" 2>/dev/null; pkill -9 -f "next start" 2>/dev/null; sleep 2; rm -rf .next node_modules/.cache; node_modules/next/dist/bin/next build > /tmp/3dmap_rebuild.log 2>&1; ec=$?; if [ $ec -ne 0 ] && grep -qiE "Cannot find module|No such file|MODULE_NOT_FOUND|ENOTEMPTY" /tmp/3dmap_rebuild.log; then echo "recovery rebuild..."; npm ci >> /tmp/3dmap_rebuild.log 2>&1; rm -rf .next; node_modules/next/dist/bin/next build >> /tmp/3dmap_rebuild.log 2>&1; ec=$?; fi; echo EXIT=$ec; pm2 start 3dmap-frontend >/dev/null 2>&1; pm2 save >/dev/null 2>&1; for i in $(seq 1 30); do c=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3000/keychains); [ "$c" = "200" ] && break; sleep 1; done; echo warmup=$c' 2>&1
        Write-Host ($rb | Out-String).Trim() -ForegroundColor DarkGray
        $check2 = (& ssh @SSH $intCmd 2>&1 | Out-String).Trim()
        Write-Host "    $check2" -ForegroundColor DarkGray
        if ($check2 -notmatch '^MATCH') {
            Write-Err "Build integrity STILL failing after clean rebuild. Log: ssh $SERVER 'cat /tmp/3dmap_rebuild.log'"
            exit 1
        }
        Write-Ok "Recovered via clean rebuild"
    } else {
        Write-Ok "Served HTML chunk exists on disk"
    }
}

# ── 8. Verify endpoints
Write-Step "Endpoint verification"
$endpoints = @(
    @{ Name = "Backend API";   Url = "http://127.0.0.1:8000/" },
    @{ Name = "Frontend /";    Url = "http://127.0.0.1:3000/" },
    @{ Name = "Frontend /create";    Url = "http://127.0.0.1:3000/create" },
    @{ Name = "Frontend /keychains"; Url = "http://127.0.0.1:3000/keychains" },
    # sitemap.xml ловить битий СЕРВЕРНИЙ білд (missing .next/server/chunks/N.js →
    # 500 на SSR-роутах, тоді як клієнтський 7c-guard цього не бачить; 2026-06-11)
    @{ Name = "Frontend sitemap.xml"; Url = "http://127.0.0.1:3000/sitemap.xml" },
    @{ Name = "Frontend /de/maps/kyiv (SSR locale)"; Url = "http://127.0.0.1:3000/de/maps/kyiv" },
    @{ Name = "Public (nginx)";   Url = "http://209.38.210.197/" }
)
function Test-Endpoints {
    $ok = $true
    foreach ($ep in $endpoints) {
        $code = & ssh @SSH "curl -s -o /dev/null -w '%{http_code}' '$($ep.Url)'" 2>&1
        if ($code -eq "200") {
            Write-Ok "$($ep.Name): $code"
        } else {
            Write-Err "$($ep.Name): $code"
            $ok = $false
        }
    }
    return $ok
}
$allOk = Test-Endpoints

# ── 8b. AUTO-HEAL: битий серверний білд (missing .next/server/chunks/N.js →
# 500 на sitemap/SSR-роутах) трапляється періодично і НЕ ловиться клієнтським
# 7c-guard. Один чистий ребілд + повторна перевірка (2026-06-11: двічі за день).
if (-not $allOk) {
    Write-Step "Endpoint check failed — attempting ONE clean rebuild (server-chunk corruption heal)"
    & ssh @SSH 'cd /opt/3dmap/frontend && pm2 stop 3dmap-frontend >/dev/null 2>&1; rm -rf .next node_modules/.cache; node_modules/next/dist/bin/next build > /tmp/3dmap_heal.log 2>&1; ec=$?; if [ $ec -ne 0 ]; then npm ci >> /tmp/3dmap_heal.log 2>&1; node_modules/next/dist/bin/next build >> /tmp/3dmap_heal.log 2>&1; ec=$?; fi; echo HEAL_EXIT=$ec; pm2 start 3dmap-frontend >/dev/null 2>&1; pm2 save >/dev/null 2>&1; for i in $(seq 1 30); do c=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3000/); [ "$c" = "200" ] && break; sleep 1; done' 2>&1 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Write-Step "Re-verifying endpoints after heal"
    $allOk = Test-Endpoints
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
