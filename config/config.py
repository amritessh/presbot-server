import os
import logging
from flask import Flask
from flask_cors import CORS
import pymongo
from pymongo import MongoClient
import boto3
from boto3.s3.transfer import TransferConfig
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

# ====================== LOGGING CONFIGURATION ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('presbot_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====================== FLASK CONFIG ======================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
app.secret_key = os.getenv('FLASK_SECRET_KEY', '')

# Session configuration for cross-domain cookies
is_production = os.getenv('FLASK_ENV') == 'production' or os.getenv(
    'ENVIRONMENT') == 'production'

if is_production:
    app.config.update(
        SESSION_COOKIE_DOMAIN='.dashlab.studio',
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax'
    )
else:
    app.config.update(
        SESSION_COOKIE_SECURE=False,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax'
    )

# CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["https://presbot.dashlab.studio", "https://dashlab.studio", "http://localhost:3002", "http://127.0.0.1:3002"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With",
            "X-CSRFToken", "X-Frame-Options", "X-XSRF-TOKEN", "x-client-timestamp", "x-request-hash"],
        "expose_headers": ["Authorization", "Content-Disposition"],
        "supports_credentials": True
    }
})

# ====================== MONGODB CONFIG ======================
MONGO_URI = "mongodb://admin:BotBox010825@localhost:27017/dash_portal_uat?authSource=admin"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.dash_portal
users_collection = db.users
presbot_originals = db.presbot_originals
presbot_voice_clones = db.presbot_voice_clones
presbot_feedback = db.presbot_feedback

# ====================== MINIO CONFIG ======================
s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name="us-east-1",
    config=boto3.session.Config(signature_version='s3v4')
)
S3_BUCKET = os.getenv("MINIO_BUCKET", "presbotclone")

# Configure multipart upload for large files
transfer_config = TransferConfig(
    multipart_threshold=5 * 1024 * 1024,  # 5 MB
    max_concurrency=10,
    multipart_chunksize=5 * 1024 * 1024,  # 5 MB chunks
    use_threads=True
)

# ====================== MODEL CONFIG ======================
WHISPER_MODEL = WhisperModel("large-v3", device="cuda")

# ====================== GRADING CONFIG ======================
LLAMA_URL = "http://localhost:5000/api/chat"
LLAMA_MODEL = "llama3.2-vision:11b"

# ====================== FILE PATHS ======================
INFER_SCRIPT = "src/f5_tts/infer/infer_cli.py"
OUTPUT_DIR = "tests"
VOICES_DIR = "voices"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)
