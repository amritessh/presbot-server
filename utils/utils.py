import os
import uuid
import tempfile
from datetime import datetime
from pydub import AudioSegment
from pydub import silence
from config import WHISPER_MODEL, s3_client, S3_BUCKET, transfer_config
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch


def transcribe_with_whisper(audio_path):
    """Transcribe audio using Whisper (faster-whisper)."""
    segments, info = WHISPER_MODEL.transcribe(audio_path,)
    transcript = "".join([segment.text for segment in segments])
    return transcript 

def get_best_12_seconds(audio_path, segment_length=12000):
    """Extract the best 12-second segment from audio."""
    audio = AudioSegment.from_file(audio_path)
    nonsilent_chunks = silence.detect_nonsilent(
        audio, min_silence_len=300, silence_thresh=audio.dBFS - 14, seek_step=1
    )
    best_segment = None
    highest_energy = float('-inf')
    for start_ms, end_ms in nonsilent_chunks:
        current_pos = start_ms
        while current_pos + segment_length <= end_ms:
            segment = audio[current_pos:current_pos + segment_length]
            energy = segment.rms
            if energy > highest_energy:
                highest_energy = energy
                best_segment = segment
            current_pos += 500
    if best_segment is None:
        best_segment = audio[:segment_length]
    return best_segment


def get_unique_filename(base_dir, base_name):
    """Generate a unique filename."""
    name, ext = os.path.splitext(base_name)
    candidate = os.path.join(base_dir, base_name)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{name}_{i}{ext}")
        i += 1
    return candidate


def upload_to_minio(file_path, username, clone_name):
    """Upload file to MinIO storage."""
    try:
        file_extension = os.path.splitext(file_path)[1]
        s3_key = f"{username}/{clone_name}{file_extension}"
        s3_client.upload_file(
            Filename=file_path,
            Bucket=S3_BUCKET,
            Key=s3_key,
            ExtraArgs={'ContentType': 'audio/wav'},
            Config=transfer_config
        )
        minio_url = f"{os.getenv('MINIO_ENDPOINT', 'http://127.0.0.1:9000')}/{S3_BUCKET}/{s3_key}"
        return minio_url, s3_key
    except Exception as e:
        print(f"Error uploading to MinIO: {str(e)}")
        raise


def process_pauses(script):
    """Process pause markers in script."""
    return script.replace('[pause]', '.')
