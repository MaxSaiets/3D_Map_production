// PM2 Ecosystem — 3dMAP Production
// Запуск: pm2 start ecosystem.config.js
// Перезапуск: pm2 restart all --update-env

module.exports = {
  apps: [
    // ─── Backend (FastAPI / uvicorn) ───────────────────────
    {
      name: '3dmap-backend',
      cwd: '/opt/3dmap/backend',
      interpreter: '/opt/3dmap/backend/venv/bin/python',
      script: '/opt/3dmap/backend/venv/bin/uvicorn',
      // Tasks are tracked in process memory, so status polling must hit the same process.
      args: 'main:app --host 127.0.0.1 --port 8000 --workers 1',
      env: {
        NODE_ENV: 'production',
        // Шлях до тимчасових файлів (видаляються після завантаження у Firebase)
        OUTPUT_DIR: '/tmp/3dmap_output',
        KEEP_LOCAL_FILES: 'false',
        // Кеш OSMnx та тайлів висот
        TERRARIUM_CACHE_DIR: '/opt/3dmap/backend/cache/terrarium',
        // Force groove booleans through Blender on production.
        BOOLEAN_BACKEND: 'blender',
        // Env файл читається через python-dotenv з .env у cwd
      },
      // Автоматичний перезапуск при крашу
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      // Логи
      out_file: '/var/log/3dmap/backend.out.log',
      error_file: '/var/log/3dmap/backend.err.log',
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      // 3D generation has memory spikes during terrain heightmap build +
      // solidification. On the 3.8GB VPS the old 3200M limit killed terrain maps
      // mid-run (graceful pm2 restart → lost task). Box has 4GB swap, so allow
      // up to 4500M: peak briefly touches swap (a little slower) instead of
      // aborting. Still far below RAM+swap (~7.8GB) so no kernel OOM.
      max_memory_restart: '4500M',
    },

    // ─── Frontend (Next.js) ────────────────────────────────
    {
      name: '3dmap-frontend',
      cwd: '/opt/3dmap/frontend',
      // Run Next directly (not `npm run start`): npm intermittently lost the
      // node_modules/.bin PATH on restart -> "sh: next: not found" crash loop.
      script: 'node_modules/next/dist/bin/next',
      args: 'start -p 3000',
      interpreter: 'node',
      env: {
        NODE_ENV: 'production',
        PORT: '3000',
        NEXT_PUBLIC_API_URL: 'http://127.0.0.1:8000',
      },
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      out_file: '/var/log/3dmap/frontend.out.log',
      error_file: '/var/log/3dmap/frontend.err.log',
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      // 512M was far too low for Next.js `next start` — pm2 killed+restarted the
      // frontend constantly, and a restart landing inside a deploy's build window
      // made it boot a half-written .next (stale-chunk flip-flop → map broke).
      // 1536M comfortably fits a steady-state next-server on this box (~4GB+swap).
      max_memory_restart: '1536M',
    },
  ],
};
