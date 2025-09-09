const path = require('path');
const cwd = process.cwd();
const base = path.basename(cwd); // e.g., 'presbot-server' or 'presbot-server-uat'
const envSuffix = /uat/i.test(base) ? 'uat' : 'prod';

module.exports = {
  apps: [
    {
      name: `presbot-api-${envSuffix}`,
      script: './start.sh',
      cwd,
      instances: 1,
      exec_mode: 'fork',
      error_file: path.join(cwd, 'logs/err.log'),
      out_file: path.join(cwd, 'logs/out.log'),
      log_file: path.join(cwd, 'logs/combined.log'),
      time: true,
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      restart_delay: 4000,
      min_uptime: '10s',
      max_restarts: 10,
      kill_timeout: 5000,
      listen_timeout: 20000
    }
  ]
};
