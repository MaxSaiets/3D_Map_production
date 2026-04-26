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
      args: 'main:app --host 127.0.0.1 --port 8000 --workers 2',
      env: {
        NODE_ENV: 'production',
        // Шлях до тимчасових файлів (видаляються після завантаження у Firebase)
        OUTPUT_DIR: '/tmp/3dmap_output',
        KEEP_LOCAL_FILES: 'false',
        // Кеш OSMnx та тайлів висот
        TERRARIUM_CACHE_DIR: '/opt/3dmap/backend/cache/terrarium',
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
      // Обмеження памʼяті — при перевищенні перезапуститься
      max_memory_restart: '1500M',
    },

    // ─── Frontend (Next.js) ────────────────────────────────
    {
      name: '3dmap-frontend',
      cwd: '/opt/3dmap/frontend',
      script: 'node_modules/.bin/next',
      args: 'start --port 3000',
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
      max_memory_restart: '512M',
    },
  ],
};
