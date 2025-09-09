#!/bin/bash

set -euo pipefail

echo "🚀 Deploying presbot-api..."

REPO_URL=${REPO_URL:-"https://github.com/amritessh/presbot-server.git"}
ENVIRONMENT=${ENVIRONMENT:-production}   # production | uat
BRANCH_NAME=${BRANCH:-release/prod}      # release/prod | release/uat

echo "📋 Using REPO_URL: $REPO_URL"
echo "📋 Using ENVIRONMENT: $ENVIRONMENT"
echo "📋 Using BRANCH: $BRANCH_NAME"

# Select target directory based on environment (allows running UAT and PROD side-by-side)
if [ "$ENVIRONMENT" = "uat" ]; then
  BASE_DIR=${BASE_DIR:-/opt/presbot-server-uat}
else
  BASE_DIR=${BASE_DIR:-/opt/presbot-server}
fi

sudo mkdir -p "$BASE_DIR"
sudo chown -R "$USER":"$USER" "$BASE_DIR"

# Bootstrap repository if missing
if [ ! -d "$BASE_DIR/.git" ]; then
  echo "📥 Cloning repository into $BASE_DIR ..."
  rm -rf "$BASE_DIR/.tmp" 2>/dev/null || true
  git clone "$REPO_URL" "$BASE_DIR"
fi

cd "$BASE_DIR"

# Ensure correct remote
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REPO_URL"
else
  git remote set-url origin "$REPO_URL"
fi

echo "📦 Using environment: ${ENVIRONMENT} | branch: ${BRANCH_NAME} | dir: ${BASE_DIR}"

echo "📦 Fetching & updating code..."
git fetch --all --prune
git checkout "${BRANCH_NAME}"
git reset --hard origin/"${BRANCH_NAME}"

# Load environment from .env if present (dotenv is also used in app)
if [ -f .env ]; then
  echo "🔐 Loading environment from .env"
  set -a
  source .env
  set +a
else
  echo "⚠️  .env not found; ensure environment variables are provided via the system"
fi

if [ ! -d ".venv" ]; then
  echo "🐍 Creating venv..."
  python3 -m venv .venv
fi

echo "🐍 Activating venv..."
source .venv/bin/activate

echo "📦 Installing Python deps..."
pip install --upgrade pip wheel
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

echo "🔧 Ensuring start.sh is executable..."
chmod +x start.sh

echo "🔧 Updating start.sh to activate venv..."
# Update start.sh to activate venv before running gunicorn
cat > start.sh << 'EOF'
#!/bin/bash

# Force bitsandbytes to load CUDA 12.8 shared library
export BITSANDBYTES_FORCE_CUDA_VERSION=128

# Activate virtual environment
source .venv/bin/activate

# Start the Gunicorn server
exec gunicorn -w 1 --timeout 720 -b 0.0.0.0:6969 api.app:app
EOF

chmod +x start.sh

echo "📁 Ensuring logs dir exists..."
mkdir -p "$BASE_DIR/logs"

echo "🛑 Restarting PM2 app..."
APP_SUFFIX=$(basename "$BASE_DIR")
APP_ENV_SUFFIX=$(echo "$APP_SUFFIX" | grep -qi uat && echo uat || echo prod)
APP_NAME="presbot-api-${APP_ENV_SUFFIX}"

sudo pm2 stop "$APP_NAME" || true
sudo pm2 delete "$APP_NAME" || true
sudo pm2 start ecosystem.config.js
sudo pm2 save

echo "🩺 Health check (local)..."
set +e
curl -sS http://127.0.0.1:6969/health | cat
set -e

echo "✅ presbot-api deployment complete."


