# CI build-and-deploy (off-server) — setup

**Why:** building the frontend on the server keeps corrupting node_modules / the npm
cache → `next/dist/bin/next` vanishes → pm2 MODULE_NOT_FOUND → 502 (~5–7 min downtime +
manual recovery, ~every other deploy). `.github/workflows/deploy.yml` builds in a clean
GitHub runner and ships the finished `.next` + `node_modules` to the server, which only
swaps + restarts. The server never runs `npm`/`next build` → that whole failure class is gone.

## What YOU must add (one-time)

GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `SERVER_IP` | `209.38.210.197` |
| `SERVER_USER` | `root` |
| `SSH_PRIVATE_KEY` | a private key whose **public** key is in the server's `/root/.ssh/authorized_keys`. Either reuse your existing deploy key, or `ssh-keygen -t ed25519 -f deploy_key`, append `deploy_key.pub` to `authorized_keys`, paste the private `deploy_key` here. |
| `FRONTEND_ENV_PRODUCTION` | the **full contents** of the server file `/opt/3dmap/frontend/.env.production` (it has `NEXT_PUBLIC_API_URL=https://monadruk.com` + all `NEXT_PUBLIC_FIREBASE_*`). Get it: `ssh root@209.38.210.197 'cat /opt/3dmap/frontend/.env.production'` and paste verbatim. These are baked into the build, so they must match prod. |

## Validate (before relying on it)

1. Push this branch / merge to `main` (the workflow is **manual-only** for now, so the push won't deploy).
2. GitHub → **Actions → "Build & Deploy (off-server)" → Run workflow** (or `gh workflow run deploy.yml`).
3. Watch it: build in CI → scp artifact → server swaps + restarts → healthcheck must end `frontend http=200`.
4. Open monadruk.com (Ctrl+Shift+R) and confirm the change is live.

## Make it automatic (after a clean validation run)

In `deploy.yml` uncomment:
```yaml
  push:
    branches: [main]
```
Then your flow becomes: edit locally → `deploy\sync.ps1 "msg" -SkipBuild` (pushes source, restarts backend) → **GitHub Actions builds the frontend off-server + ships it**. No more manual `reland.sh`, no more corruption lottery.

## Fallback (until CI is validated)

The current path still works: `deploy\sync.ps1 "msg" -SkipBuild` then `ssh root@… 'nohup bash /tmp/reland.sh …'`. `reland.sh` now self-heals (npm cache clean + reinstall) if corruption hits, but still costs downtime — which is exactly what this CI removes.

## Notes / trade-offs

- Artifact = `.next` + `node_modules` tarball (~150–250 MB). CI run ~5–8 min; **site downtime only ~1–2 min** (the swap+restart), vs 2–7 min server build + corruption risk today.
- CI runner (`ubuntu-latest`, x64) matches the server, so native binaries (swc/sharp) are compatible.
- `concurrency.group: deploy-main` prevents overlapping deploys.
- Backend is still deployed via the SSH step (`git pull` + `pip install` + restart).
