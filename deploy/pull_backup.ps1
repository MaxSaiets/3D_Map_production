# pull_backup.ps1 — ХОСТ-ПК (де крутиться VM): щодня о 00:30 забирає повний бекап
# monadruk.com з VM на C:\monadruk-backup, ПЕРЕЗАПИСУЮЧИ попередній. Якщо архів на VM
# не змінився (той самий sha256) — нічого не качає (економія диска й часу).
# Завдання планувальника: monadruk-backup-pull (SYSTEM). Лог: C:\monadruk-backup\pull.log
# Ключ C:\monadruk-backup\vm_backup_key на VM дозволяє ЛИШЕ «sha» і «get»
# (forced command /usr/local/bin/monadruk-backup-serve) — нічого іншого ним не зробити.

$ErrorActionPreference = 'Stop'
$Dir     = 'C:\monadruk-backup'
$Key     = "$Dir\vm_backup_key"
$Known   = "$Dir\known_hosts"
$Target  = "$Dir\monadruk-full.tar.gz"
$ShaFile = "$Dir\monadruk-full.sha256"
$Log     = "$Dir\pull.log"
$VM      = 'deploy@192.168.0.4'
$Ssh     = 'C:\Program Files\OpenSSH-Win64\ssh.exe'
# шляхи без пробілів → без лапок (cmd /c ламав вкладені лапки й ssh зависав)
$SshArgs = @('-i', $Key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15',
             '-o', 'StrictHostKeyChecking=accept-new', '-o', "UserKnownHostsFile=$Known", $VM)

function Log($m) {
    Add-Content -LiteralPath $Log -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $m) -Encoding UTF8
    $lines = Get-Content -LiteralPath $Log -Encoding UTF8
    if ($lines.Count -gt 400) { $lines | Select-Object -Last 300 | Set-Content -LiteralPath $Log -Encoding UTF8 }
}

try {
    # stderr ssh — у файл: у PowerShell 5.1 з ErrorActionPreference=Stop будь-який рядок
    # у stderr нативної програми (навіть попередження) обриває скрипт
    $ErrorActionPreference = 'Continue'
    $remote = (& $Ssh @SshArgs 'sha' 2>"$Dir\ssh_err.txt" | Out-String).Trim()
    $ErrorActionPreference = 'Stop'
    if (-not $remote) { Log "ERROR: VM не відповіла (ssh sha): $((Get-Content "$Dir\ssh_err.txt" -Raw) -replace '\s+',' ')"; exit 1 }
    $remoteSha = ($remote -split '\s+')[0].Trim().ToLower()
    if ($remoteSha -notmatch '^[0-9a-f]{64}$') { Log "ERROR: дивна відповідь VM: $remote"; exit 1 }

    $localSha = if (Test-Path -LiteralPath $ShaFile) { (Get-Content -LiteralPath $ShaFile -Raw).Trim().ToLower() } else { '' }
    if ($localSha -eq $remoteSha -and (Test-Path -LiteralPath $Target)) {
        Log "OK: бекап не змінився ($($remoteSha.Substring(0,12))) — пропускаю"
        exit 0
    }

    $part = "$Target.part"
    if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
    # Start-Process пише stdout у файл побайтово (PowerShell `>` псує бінарні дані)
    $p = Start-Process -FilePath $Ssh -ArgumentList (($SshArgs + 'get') -join ' ') -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $part -RedirectStandardError "$Dir\ssh_err.txt"
    if ($p.ExitCode -ne 0) { Log "WARN: ssh get exit=$($p.ExitCode)" }
    if (-not (Test-Path -LiteralPath $part)) { Log "ERROR: файл не отримано"; exit 1 }
    $gotSha = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash.ToLower()
    if ($gotSha -ne $remoteSha) {
        Remove-Item -LiteralPath $part -Force
        Log "ERROR: контрольна сума не збіглась (очікувано $($remoteSha.Substring(0,12)), отримано $($gotSha.Substring(0,12))) — старий бекап лишаю"
        exit 1
    }
    Move-Item -LiteralPath $part -Destination $Target -Force
    Set-Content -LiteralPath $ShaFile -Value $remoteSha -Encoding ASCII
    $mb = [math]::Round((Get-Item -LiteralPath $Target).Length / 1MB)
    Log "OK: оновлено бекап ${mb} МБ ($($remoteSha.Substring(0,12)))"
    exit 0
} catch {
    Log "ERROR: $($_.Exception.Message)"
    exit 1
}
