import logging
from flask import request, jsonify, send_file, session, redirect
import subprocess
import os
import uuid
import json
import jwt
from functools import wraps
from datetime import datetime
import tempfile
from bson import ObjectId
import torch

# Import from our organized modules
from config import (
    app, logger, users_collection, presbot_originals, presbot_voice_clones,
    presbot_feedback, s3_client, S3_BUCKET, transfer_config, INFER_SCRIPT,
    OUTPUT_DIR, VOICES_DIR, LLAMA_URL, LLAMA_MODEL, db, is_production
)
from utils.utils import (
    transcribe_with_whisper, get_best_12_seconds, get_unique_filename,
    upload_to_minio, process_pauses
)
from agents.audio_grading_agent import VoiceGradingAgent
from agents.script_grading_agent import ScriptGradingAgent

# Pydantic models
from models.models import (
    OriginalVoiceSample, VoiceClone, UserRegistration, CloneCreation,
    CloneUpdate, GradeRequest, GradingFeedback, AudioGradingResult,
    ScriptGradingResult, GradingResponse
)

# Initialize grading agents
voiceclone_grading_agent = VoiceGradingAgent(
    llama_url="http://localhost:5000/api/generate",
    llama_model=LLAMA_MODEL
)
script_grading_agent = ScriptGradingAgent(
    llama_url="http://localhost:5000/api/chat",
    llama_model=LLAMA_MODEL
)

# ====================== AUTHENTICATION ======================


@app.route('/auth', methods=['POST'])
def authenticate_from_dash_portal():
    """Receives secure authentication from Dash Portal"""
    try:
        token = request.form.get('token')
        if not token:
            logger.error("No authentication token provided")
            return "No authentication token provided", 400

        jwt_secret = os.getenv('JWT_SECRET')
        if not jwt_secret:
            logger.error("JWT_SECRET environment variable not set")
            return "Server configuration error", 500

        try:
            payload = jwt.decode(
                token,
                jwt_secret,
                algorithms=['HS256'],
                audience='presbot'
            )
            logger.info(f"JWT payload decoded successfully: {payload}")
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return "Login expired. Please return to Dash Portal.", 401
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid JWT token: {e}")
            return "Invalid login token. Please return to Dash Portal.", 401
        except Exception as e:
            logger.error(f"JWT decode error: {e}")
            return "Invalid token format", 400

        # Validate required fields
        required_fields = ['id', 'email', 'service']
        for field in required_fields:
            if field not in payload:
                logger.error(f"Missing required field in JWT payload: {field}")
                return "Invalid token format", 400

        user_id = payload['id']
        user_email = payload['email']
        service = payload['service']

        if service != 'presbot':
            logger.warning(f"Invalid service in token: {service}")
            return "Invalid service token", 400

        try:
            user_object_id = ObjectId(user_id)
        except Exception as e:
            logger.error(f"Invalid user ID format: {user_id}, error: {e}")
            return "Invalid user ID format", 400

        user = db.users.find_one({"_id": user_object_id})

        if not user:
            logger.warning(f"User not found in database: {user_id}")
            return "User not found", 404

        # Update last login
        db.users.update_one(
            {"_id": user_object_id},
            {"$set": {"last_login": datetime.utcnow()}}
        )

        # Set session
        session['user_id'] = user_id
        session['email'] = user_email
        session['authenticated'] = True
        session.permanent = True

        logger.info(f"User authenticated successfully: {user_email}")

        redirect_url = 'http://localhost:3002' if not is_production else 'https://presbot.dashlab.studio'
        return redirect(redirect_url)

    except Exception as e:
        logger.error(f"Unexpected authentication error: {e}")
        return "Authentication failed. Please try again.", 500


def require_auth(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect('/login-required')
        return f(*args, **kwargs)
    return decorated_function


def require_api_auth(f):
    """Decorator to require authentication for API routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return jsonify({
                'error': 'Authentication required',
                'message': 'Please log in through Dash Portal to access this endpoint',
                'authenticated': False
            }), 401
        return f(*args, **kwargs)
    return decorated_function

# ====================== GRADING ENDPOINTS ======================


@app.route('/grade', methods=['POST'])
# @require_api_auth  # Commented out for testing
def grade_audio():
    """Grade audio with real-time feedback"""
    try:
        form = request.form if request.form else request.json

        presentation_type = form.get("presentation_type", "Class Presentation")
        audience = form.get("audience", "Professors/Teachers")
        goals = form.getlist("goals") if hasattr(
            form, "getlist") else form.get("goals", [])
        if isinstance(goals, str):
            goals = [g.strip() for g in goals.split(",") if g.strip()]
        custom_goals = form.get("custom_goals", "")
        user_id = form.get("user_id")  # Add this
        original_id = form.get("original_id")  # Add this

        # Handle both file upload and MinIO URL
        audio_path = None

        # Check if audio_url is provided (MinIO URL from upload step)
        if 'audio_url' in form:
            audio_url = form.get('audio_url')
            # Download file from MinIO URL
            import requests
            response = requests.get(audio_url)
            if response.status_code == 200:
                unique_id = str(uuid.uuid4())
                audio_path = f"{VOICES_DIR}/{unique_id}_grade.wav"
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
            else:
                return jsonify({"error": "Failed to download audio from MinIO URL"}), 400

        # Check if audio file is uploaded directly (fallback)
        elif 'audio' in request.files:
            audio = request.files['audio']
            unique_id = str(uuid.uuid4())
            audio_path = f"{VOICES_DIR}/{unique_id}_grade.wav"
            audio.save(audio_path)

        else:
            return jsonify({"error": "No audio file or MinIO URL provided"}), 400

        if not audio_path:
            return jsonify({"error": "Failed to process audio"}), 400

        transcript = transcribe_with_whisper(audio_path)
        print("AUDIO PATH:", audio_path)
        print("WHISPER TRANSCRIPT:", transcript)

        # Phase 1: VoiceClone Audio grading
        print("Starting VoiceClone audio grading...")
        audio_results = voiceclone_grading_agent.grade_audio_delivery(
            audio_path, presentation_type, audience, goals, custom_goals, transcript
        )
        print("VoiceClone audio grading completed.")

        refined_script = audio_results.get("script", "")
        grade_result = audio_results
        score = grade_result.get("score", 0)
        improvement_needed = score < 4

        # Phase 2: Script grading (synchronous)
        print("Starting script grading...")
        script_results = script_grading_agent.grade_script_content(
            original_script=transcript,
            refined_script=refined_script,
            presentation_type=presentation_type,
            audience=audience,
            goals=goals,
            custom_goals=custom_goals
        )
        print("Script grading completed.")

        # Store feedback
        user_id = form.get("user_id", None)
        original_id = form.get("original_id", None)
        feedback_doc = {
            "grading_id": unique_id,
            "user_id": user_id,
            "original_id": original_id,
            "status": "completed",
            "context": {
                "presentation_type": presentation_type,
                "audience": audience,
                "goals": goals,
                "custom_goals": custom_goals
            },
            "original_transcript": transcript,
            "refined_transcript": grade_result.get("script", ""),
            "score": score,
            "overall_feedback": grade_result.get("overall_feedback"),
            "rubric_details": grade_result.get("rubric_details", []),
            "key_moments": grade_result.get("key_moments", [])[:4],
            "improvement_needed": improvement_needed,
            "script_results": script_results,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "completed_at": datetime.utcnow()
        }
        presbot_feedback.insert_one(feedback_doc)

        try:
            os.remove(audio_path)
        except Exception as e:
            print(
                f"Warning: Could not delete temp audio file {audio_path}: {e}")

        return jsonify({
            "grading_id": unique_id,
            "status": "completed",
            "script_analysis_pending": False,
            "audio_grading": {
                "score": audio_results.get("score", 0),
                "feedback": audio_results.get("overall_feedback", {}),
                "key_metrics": audio_results.get("key_metrics", {}),
                "original_transcript": transcript,
                "refined_transcript": process_pauses(grade_result.get("script", "")),
                "categories_evaluated": list(audio_results.get("voiceclone_raw_output", {}).get("rubric", {}).keys()),
                "improvement_needed": improvement_needed,
            },
            "script_grading": script_results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====================== VOICE CLONE ENDPOINTS ======================


@app.route('/update_clone', methods=['PUT'])
@require_api_auth
def update_voice_clone():
    """Update voice clone details"""
    try:
        update_data = CloneUpdate(**request.get_json())

        # Validate user exists
        user_doc = users_collection.find_one(
            {"_id": ObjectId(update_data.user_id)})
        if not user_doc:
            return jsonify({"error": "User not found"}), 404

        # Find clone in presbot_voice_clones collection
        clone_doc = presbot_voice_clones.find_one({
            "user_id": update_data.user_id,
            "clone_name": update_data.old_clone_name
        })

        if not clone_doc:
            return jsonify({"error": "Voice clone not found"}), 404

        # Update the actual clone document in presbot_voice_clones collection
        update_result = presbot_voice_clones.update_one(
            {"_id": clone_doc["_id"]},
            {
                "$set": {
                    "clone_name": update_data.new_clone_name,
                    "note": update_data.note,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        if update_result.modified_count == 0:
            return jsonify({"error": "Failed to update clone"}), 500

        # Get the updated clone document
        updated_clone_doc = presbot_voice_clones.find_one(
            {"_id": clone_doc["_id"]})

        # Prepare response data
        updated_clone_data = {
            "clone_id": str(updated_clone_doc["_id"]),
            "clone_name": updated_clone_doc["clone_name"],
            "clone_s3_key": updated_clone_doc["clone_s3_key"],
            "clone_minio_url": updated_clone_doc["clone_minio_url"],
            "created_at": updated_clone_doc["created_at"],
            "updated_at": updated_clone_doc.get("updated_at"),
            "gen_text": updated_clone_doc["gen_text"],
            "ref_text": updated_clone_doc["ref_text"],
            "note": updated_clone_doc["note"],
            "config": updated_clone_doc.get("config", {})
        }

        logger.info(
            f"Clone updated successfully: {update_data.old_clone_name} -> {update_data.new_clone_name}")

        return jsonify({
            "message": "Voice clone updated successfully",
            "user_id": update_data.user_id,
            "old_clone_name": update_data.old_clone_name,
            "updated_clone": updated_clone_data
        }), 200

    except ValueError as e:
        logger.error(f"Validation error in update_clone: {str(e)}")
        return jsonify({"error": f"Validation error: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error updating clone: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/clone', methods=['POST'])
@require_api_auth
def clone_voice():
    """Create voice clone"""
    try:
        logger.info("=== CLONE REQUEST STARTED ===")
        ref_audio = request.form.get('ref_audio')
        temp_audio_path = None

        if ref_audio and (ref_audio.startswith('http://') or ref_audio.startswith('https://')):
            import requests
            response = requests.get(ref_audio)
            if response.status_code != 200:
                return jsonify({"error": "Failed to download audio from MinIO URL"}), 400
            with tempfile.NamedTemporaryFile(suffix="_original_ref.wav", delete=False) as temp_orig:
                temp_orig.write(response.content)
                temp_audio_path = temp_orig.name
        elif 'ref_audio' in request.files:
            file = request.files['ref_audio']
            with tempfile.NamedTemporaryFile(suffix="_original_ref.wav", delete=False) as temp_orig:
                file.save(temp_orig.name)
                temp_audio_path = temp_orig.name
        else:
            return jsonify({"error": "ref_audio (MinIO URL or file) is required"}), 400

        config_fields = {}
        for key in ["speed", "nfe_step", "cross_fade_duration", "pitch_shift", "seed"]:
            if key in request.form:
                config_fields[key] = request.form.get(key)

        try:
            form_data = CloneCreation(
                user_id=request.form.get('user_id', '').strip(),
                clone_name=request.form.get('clone_name', '').strip(),
                gen_text=request.form.get('gen_text', '').strip(),
                # This will be ignored, we use transcribed text instead
                ref_text=request.form.get('ref_text', '').strip(),
                note=request.form.get('note', '').strip(),
                speed=float(request.form.get('speed', 1.0)),
                nfe_step=int(request.form.get('nfe_step', 32)),
                cross_fade_duration=float(
                    request.form.get('cross_fade_duration', 0.15))
            )
            original_id = request.form.get('original_id', '').strip()
            pitch_shift = float(request.form.get('pitch_shift', 0.0))
            seed = int(request.form.get('seed', 0))
        except ValueError as e:
            if temp_audio_path:
                os.remove(temp_audio_path)
            return jsonify({"error": f"Validation error: {str(e)}"}), 400

        if not form_data.user_id or not original_id:
            if temp_audio_path:
                os.remove(temp_audio_path)
            return jsonify({"error": "user_id and original_id are required"}), 400

        user_doc = users_collection.find_one(
            {"_id": ObjectId(form_data.user_id)})
        if not user_doc:
            if temp_audio_path:
                os.remove(temp_audio_path)
            return jsonify({"error": "User not found. Please register first."}), 404

        unique_id = str(uuid.uuid4())

        # Process audio
        from pydub import AudioSegment
        original_audio = AudioSegment.from_file(temp_audio_path)
        logger.info(
            f"Original audio - duration: {len(original_audio)/1000:.2f} seconds")
        logger.info(
            f"Original audio - sample_rate: {original_audio.frame_rate}Hz")

        best_segment = get_best_12_seconds(temp_audio_path)
        logger.info(
            f"Best 12s segment - duration: {len(best_segment)/1000:.2f} seconds")
        logger.info(f"Best 12s segment - energy: {best_segment.rms:.2f}")

        with tempfile.NamedTemporaryFile(suffix="_ref.wav", delete=False) as temp_clip:
            best_segment.export(temp_clip.name, format="wav")
            clipped_audio_path = temp_clip.name

        # Get the full transcript from the grading process
        # Always transcribe the 12-second clipped audio to get reference text
        # Note: We intentionally ignore any UI-provided full_transcript for ref_text
        logger.info(
            "Transcribing 12-second clipped audio with Whisper for reference text...")
        ref_text = transcribe_with_whisper(clipped_audio_path)
        logger.info(f"Whisper transcription of 12s clip: '{ref_text}'")
        logger.info(f"Using 12s clip transcript for F5-TTS: '{ref_text}'")

        # Use the transcript as ref_text (either full or 12s clip)
        # Pass ref_text content directly instead of via file
        logger.info(f"Passing ref_text directly to F5-TTS: '{ref_text}'")

        output_filename = f"{form_data.clone_name}_{unique_id}.wav"
        with tempfile.NamedTemporaryFile(suffix="_clone.wav", delete=False) as temp_out:
            output_path = temp_out.name

        command = [
            "python3", INFER_SCRIPT,
            "--gen_text", form_data.gen_text,
            "--ref_audio", clipped_audio_path,
            "--ref_text", ref_text,
            "--output_dir", os.path.dirname(output_path),
            "--output_file", os.path.basename(output_path),
            "--speed", str(form_data.speed),
            "--nfe_step", str(form_data.nfe_step),
            "--cross_fade_duration", str(form_data.cross_fade_duration)
        ]

        logger.info("=== F5-TTS COMMAND EXECUTION ===")
        logger.info(f"Command: {' '.join(command)}")
        logger.info(f"Expected output path: {output_path}")

        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(form_data.gen_text.encode("utf-8"))
        ref_audio_len_seconds = len(best_segment) / 1000
        expected_duration = ref_audio_len_seconds * \
            (gen_text_len / ref_text_len) / form_data.speed
        logger.info(f"Expected output duration: {expected_duration:.2f}s")

        try:
            result = subprocess.run(command, check=True,
                                    capture_output=True, text=True)
            logger.info(f"F5-TTS execution completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"F5-TTS command failed with exit code {e.returncode}")
            logger.error(f"F5-TTS stdout: {e.stdout}")
            logger.error(f"F5-TTS stderr: {e.stderr}")
            raise

        if os.path.exists(output_path):
            output_audio = AudioSegment.from_file(output_path)
            actual_duration = len(output_audio) / 1000
            logger.info(
                f"Generated audio - duration: {actual_duration:.2f} seconds")
            logger.info(
                f"Generated audio - sample_rate: {output_audio.frame_rate}Hz")
            logger.info(
                f"Duration difference from expected: {actual_duration - expected_duration:.2f}s")
        else:
            logger.error(f"Output file not found: {output_path}")

        # Upload to MinIO
        clone_s3_key = f"{form_data.user_id}/clones/{form_data.clone_name}_{unique_id}.wav"
        try:
            s3_client.upload_file(
                Filename=output_path,
                Bucket=S3_BUCKET,
                Key=clone_s3_key,
                ExtraArgs={'ContentType': 'audio/wav'},
                Config=transfer_config
            )
            clone_minio_url = f"{os.getenv('MINIO_ENDPOINT', 'http://127.0.0.1:9000')}/{S3_BUCKET}/{clone_s3_key}"
            logger.info(
                f"Successfully uploaded clone to MinIO: {clone_minio_url}")
        except Exception as e:
            logger.error(f"Failed to upload clone to MinIO: {str(e)}")
            # Cleanup temp files
            for f in [temp_audio_path, clipped_audio_path, output_path]:
                try:
                    os.remove(f)
                except Exception as cleanup_e:
                    logger.warning(
                        f"Could not delete temp file {f}: {cleanup_e}")
            return jsonify({"error": f"Failed to upload clone to storage: {str(e)}"}), 500

        # Store metadata
        voice_clone_doc = {
            "user_id": form_data.user_id,
            "clone_name": form_data.clone_name,
            "original_id": original_id,
            "clone_s3_key": clone_s3_key,
            "clone_minio_url": clone_minio_url,
            "created_at": datetime.utcnow(),
            "gen_text": form_data.gen_text,
            "ref_text": ref_text,
            "note": form_data.note,
            "config": config_fields
        }

        try:
            result = presbot_voice_clones.insert_one(voice_clone_doc)
            presbot_originals.update_one({"_id": ObjectId(original_id)}, {
                "$push": {"clone_ids": result.inserted_id}})
            logger.info(
                f"Successfully stored clone metadata in database: {str(result.inserted_id)}")
        except Exception as e:
            logger.error(
                f"Failed to store clone metadata in database: {str(e)}")
            # Cleanup temp files
            for f in [temp_audio_path, clipped_audio_path, output_path]:
                try:
                    os.remove(f)
                except Exception as cleanup_e:
                    logger.warning(
                        f"Could not delete temp file {f}: {cleanup_e}")
            return jsonify({"error": f"Failed to store clone metadata: {str(e)}"}), 500

        # Cleanup
        for f in [temp_audio_path, clipped_audio_path, output_path]:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: Could not delete temp file {f}: {e}")

        logger.info("=== CLONE REQUEST COMPLETED SUCCESSFULLY ===")
        logger.info(f"Clone uploaded to MinIO: {clone_minio_url}")
        logger.info(f"Clone ID: {str(result.inserted_id)}")

        return jsonify({
            "message": "Voice clone created successfully",
            "clone_name": form_data.clone_name,
            "clone_minio_url": clone_minio_url,
            "user_id": form_data.user_id,
            "clone_id": str(result.inserted_id),
            "config": config_fields
        }), 200
    except Exception as e:
        logger.error(f"=== CLONE REQUEST FAILED ===")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# ====================== OTHER ENDPOINTS ======================


@app.route('/upload_original', methods=['POST'])
@require_api_auth
def upload_original_voice():
    """Upload original voice sample"""
    try:
        if 'ref_audio' not in request.files:
            return jsonify({"error": "ref_audio file is required"}), 400

        ref_audio = request.files['ref_audio']
        note = request.form.get('note', '').strip()
        user_id = request.form.get('user_id', '').strip()

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        user_doc = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user_doc:
            return jsonify({"error": "User not found in users collection"}), 404

        timestamp = int(datetime.utcnow().timestamp())
        file_extension = '.wav'
        generated_name = f"original_{timestamp}{file_extension}"
        s3_key = f"{user_id}/originals/{generated_name}"

        try:
            s3_client.upload_fileobj(
                ref_audio,
                S3_BUCKET,
                s3_key,
                ExtraArgs={'ContentType': 'audio/wav'},
                Config=transfer_config
            )
            minio_url = f"{os.getenv('MINIO_ENDPOINT', 'http://127.0.0.1:9000')}/{S3_BUCKET}/{s3_key}"
            logger.info(
                f"Successfully uploaded original to MinIO: {minio_url}")
        except Exception as e:
            logger.error(f"Failed to upload original to MinIO: {str(e)}")
            return jsonify({"error": f"Failed to upload original to storage: {str(e)}"}), 500

        original_doc = {
            "user_id": user_id,
            "original_name": generated_name,
            "original_s3_key": s3_key,
            "original_minio_url": minio_url,
            "created_at": datetime.utcnow(),
            "note": note,
            "clone_ids": []
        }

        try:
            result = presbot_originals.insert_one(original_doc)
            logger.info(
                f"Successfully stored original metadata in database: {str(result.inserted_id)}")
        except Exception as e:
            logger.error(
                f"Failed to store original metadata in database: {str(e)}")
            return jsonify({"error": f"Failed to store original metadata: {str(e)}"}), 500

        return jsonify({
            "message": "Original voice sample uploaded successfully",
            "minio_url": minio_url,
            "s3_key": s3_key,
            "user_id": user_id,
            "original_name": generated_name,
            "original_id": str(result.inserted_id)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/list_originals/<user_id>', methods=['GET'])
@require_api_auth
def list_originals(user_id):
    """List original voice samples for user"""
    try:
        originals_cursor = presbot_originals.find({"user_id": user_id})
        originals = []

        for doc in originals_cursor:
            doc["original_id"] = str(doc["_id"])
            doc.pop("_id", None)

            if "clone_ids" in doc:
                doc["clone_ids"] = [str(cid) for cid in doc["clone_ids"]]

            clones = []
            for clone_id in doc.get("clone_ids", []):
                clone_doc = presbot_voice_clones.find_one(
                    {"_id": ObjectId(clone_id)})
                if clone_doc:
                    clone_doc["clone_id"] = str(clone_doc["_id"])
                    clone_doc.pop("_id", None)
                    clones.append(clone_doc)
            doc["clones"] = clones
            originals.append(doc)

        return jsonify({
            "user_id": user_id,
            "total_originals": len(originals),
            "originals": originals
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/list_clones_for_original/<original_id>', methods=['GET'])
@require_api_auth
def list_clones_for_original(original_id):
    """List clones for a specific original"""
    try:
        original = presbot_originals.find_one({"_id": ObjectId(original_id)})
        if not original:
            return jsonify({"error": "Original not found"}), 404

        clone_ids = original.get("clone_ids", [])
        clones = []

        for clone_id in clone_ids:
            clone_doc = presbot_voice_clones.find_one({"_id": clone_id})
            if clone_doc:
                clone_doc["clone_id"] = str(clone_doc["_id"])
                clone_doc.pop("_id", None)
                clones.append(clone_doc)

        return jsonify({
            "original_id": original_id,
            "total_clones": len(clones),
            "clones": clones
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analysis_results/<original_id>', methods=['GET'])
@require_api_auth
def get_analysis_results(original_id):
    """Get analysis results for an original"""
    try:
        feedback_doc = presbot_feedback.find_one(
            {"original_id": original_id},
            sort=[("created_at", -1)]
        )
        if not feedback_doc:
            return jsonify({"error": "No analysis results found for this original"}), 404

        original_doc = presbot_originals.find_one(
            {"_id": ObjectId(original_id)})
        if not original_doc:
            return jsonify({"error": "Original not found"}), 404

        enhanced_clone = None
        available_clones = []

        if original_doc.get("clone_ids"):
            clones_cursor = presbot_voice_clones.find({
                "_id": {"$in": original_doc["clone_ids"]}
            }).sort("created_at", -1)

            for clone in clones_cursor:
                clone["clone_id"] = str(clone["_id"])
                clone.pop("_id", None)
                available_clones.append(clone)

            if available_clones:
                enhanced_clone = available_clones[0]

        response = {
            "analysis": {
                "feedback_id": str(feedback_doc["_id"]),
                "original_id": original_id,
                "score": feedback_doc.get("score", 0),
                "overall_feedback": feedback_doc.get("overall_feedback", ""),
                "improvement_needed": feedback_doc.get("improvement_needed", False),
                "created_at": feedback_doc.get("created_at"),
                "context": feedback_doc.get("context", {})
            },
            "transcripts": {
                "original": feedback_doc.get("original_transcript", ""),
                "refined": feedback_doc.get("refined_transcript", "")
            },
            "audio": {
                "original_url": original_doc.get("original_minio_url", ""),
                "enhanced_url": enhanced_clone.get("clone_minio_url", "") if enhanced_clone else "",
                "enhanced_clone_id": enhanced_clone.get("clone_id", "") if enhanced_clone else "",
                "total_clones": len(available_clones)
            },
            "performance": {
                "rubric_details": feedback_doc.get("rubric_details", []),
                "key_moments": feedback_doc.get("key_moments", [])
            },
            "enhanced_script": {
                "text": enhanced_clone.get("gen_text", "") if enhanced_clone else "",
                "config": enhanced_clone.get("config", {}) if enhanced_clone else {}
            },
            "available_clones": available_clones
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "message": "Presbot server is running",
        "environment": {
            "production": is_production,
            "jwt_secret_set": bool(os.getenv('JWT_SECRET')),
            "mongo_connected": True,  # Will be False if connection fails
            "minio_connected": True,  # Will be False if connection fails
            "session_authenticated": session.get('authenticated', False)
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    # Test MongoDB connection
    try:
        db.command('ping')
        health_status["environment"]["mongo_connected"] = True
    except Exception as e:
        health_status["environment"]["mongo_connected"] = False
        health_status["mongo_error"] = str(e)
        health_status["status"] = "degraded"

    # Test MinIO connection
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        health_status["environment"]["minio_connected"] = True
    except Exception as e:
        health_status["environment"]["minio_connected"] = False
        health_status["minio_error"] = str(e)
        health_status["status"] = "degraded"

    return jsonify(health_status), 200


@app.route('/logout')
def logout():
    """Clear session and logout"""
    session.clear()
    return redirect('https://dashlash.studio')


@app.route('/api/session-status', methods=['GET'])
def get_session_status():
    """API endpoint to check current session status"""
    if 'authenticated' in session and session['authenticated']:
        return jsonify({
            'authenticated': True,
            'user_id': session.get('user_id'),
            'email': session.get('email')
        })
    return jsonify({'authenticated': False}), 401


@app.route('/api/gpu-cleanup', methods=['POST'])
@require_api_auth
def gpu_cleanup():
    """Force GPU cleanup for the Qwen audio grading agent"""
    try:
        # Unload Qwen model
        qwen_audio_grading_agent.unload_model()

        # Also clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify({
            "status": "success",
            "message": "GPU cleanup completed successfully",
            "model_status": {
                "qwen_loaded": qwen_audio_grading_agent.model_loaded
            }
        })
    except Exception as e:
        logger.error(f"GPU cleanup failed: {e}")
        return jsonify({
            "status": "error",
            "message": f"GPU cleanup failed: {str(e)}"
        }), 500


@app.route('/api/agent-status', methods=['GET'])
@require_api_auth
def agent_status():
    """Get detailed status of all grading agents"""
    try:
        qwen_status = qwen_audio_grading_agent.health_check()

        return jsonify({
            "status": "success",
            "agents": {
                "qwen_audio": qwen_status
            },
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Agent status check failed: {e}")
        return jsonify({
            "status": "error",
            "message": f"Agent status check failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=6969, debug=True)
