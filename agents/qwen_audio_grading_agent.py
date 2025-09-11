"""
Qwen2-Audio based audio grading agent for direct speech analysis.
This agent can actually listen to audio and evaluate delivery characteristics.
"""

import json
import torch
import torchaudio
import librosa
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import tempfile
import os
import logging
import time
import threading
import gc

logger = logging.getLogger(__name__)


class QwenAudioGradingAgent:
    """
    Agent that directly analyzes audio using Qwen2-Audio for speech delivery assessment.
    This agent can hear and evaluate actual audio characteristics like tone, pacing, confidence, etc.
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 rubric_path: str = "config/rubric.json",
                 torch_dtype=torch.bfloat16,
                 idle_timeout: float = 300.0):  # 5 minutes default
        """
        Initialize the Qwen2-Audio grading agent.

        Args:
            model_name: HuggingFace model identifier for Qwen2-Audio
            device: Device to run the model on (cuda/cpu)
            rubric_path: Path to the grading rubric JSON file
            torch_dtype: PyTorch data type for model (bfloat16 optimal for Blackwell)
            idle_timeout: Time in seconds before unloading model when idle (default: 300s/5min)
        """
        self.device = device
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.rubric_path = rubric_path
        self.idle_timeout = idle_timeout

        # Load rubric
        with open(self.rubric_path, 'r') as f:
            self.rubric = json.load(f)

        # Initialize model and processor (lazy loading)
        self.model = None
        self.processor = None
        self.model_loaded = False

        # Idle management
        self.last_used = time.time()
        self.cleanup_timer = None
        self._cleanup_lock = threading.Lock()

        logger.info(
            f"QwenAudioGradingAgent initialized. Model: {model_name}, Device: {device}, Idle timeout: {idle_timeout}s")

    def _load_model(self):
        """Lazy load the Qwen2-Audio model and processor."""
        if self.model_loaded:
            self._update_last_used()
            return

        try:
            logger.info(f"Loading Qwen2-Audio model: {self.model_name}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model with optimizations for Blackwell GPU
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if self.device == "cuda" else None,
            )

            self.model_loaded = True
            self._update_last_used()
            self._schedule_cleanup()
            logger.info("✅ Qwen2-Audio model loaded successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to load Qwen2-Audio model: {e}")
            raise Exception(f"Model loading failed: {e}")

    def _update_last_used(self):
        """Update the last used timestamp and reset cleanup timer."""
        self.last_used = time.time()
        self._schedule_cleanup()

    def _schedule_cleanup(self):
        """Schedule automatic model cleanup after idle timeout."""
        with self._cleanup_lock:
            # Cancel existing timer
            if self.cleanup_timer:
                self.cleanup_timer.cancel()

            # Schedule new cleanup
            self.cleanup_timer = threading.Timer(
                self.idle_timeout, self._auto_cleanup)
            self.cleanup_timer.start()

    def _auto_cleanup(self):
        """Automatically cleanup model if it's been idle."""
        with self._cleanup_lock:
            current_time = time.time()
            time_since_last_use = current_time - self.last_used

            if time_since_last_use >= self.idle_timeout and self.model_loaded:
                logger.info(
                    f"Auto-cleaning up Qwen model after {time_since_last_use:.1f}s idle time")
                self.unload_model()

    def unload_model(self):
        """
        Safely unload the model and processor from GPU memory.
        This frees up GPU resources when the model is not in use.
        """
        with self._cleanup_lock:
            if not self.model_loaded:
                logger.info("Model already unloaded")
                return

            try:
                logger.info("🧹 Unloading Qwen2-Audio model from GPU...")

                # Cancel cleanup timer
                if self.cleanup_timer:
                    self.cleanup_timer.cancel()
                    self.cleanup_timer = None

                # Move model to CPU first if it exists and is on GPU
                if self.model is not None:
                    if hasattr(self.model, 'cpu'):
                        self.model.cpu()
                    del self.model
                    self.model = None

                # Clear processor
                if self.processor is not None:
                    del self.processor
                    self.processor = None

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Force garbage collection
                gc.collect()

                self.model_loaded = False
                logger.info("✅ Qwen2-Audio model unloaded successfully!")

                # Log GPU memory status if available
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    cached = torch.cuda.memory_reserved() / (1024**3)  # GB
                    logger.info(
                        f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

            except Exception as e:
                logger.error(f"❌ Error during model cleanup: {e}")
                # Still mark as unloaded to prevent stuck state
                self.model_loaded = False

    def _preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file for optimal Qwen2-Audio processing.

        Args:
            audio_path: Path to the audio file

        Returns:
            Path to processed audio file
        """
        try:
            # Load audio
            # Qwen2-Audio prefers 16kHz
            audio, sr = librosa.load(audio_path, sr=16000)

            # Normalize audio
            audio = librosa.util.normalize(audio)

            # Remove silence from beginning and end
            audio, _ = librosa.effects.trim(audio, top_db=20)

            # Limit to reasonable length (e.g., 5 minutes max for performance)
            max_length = 5 * 60 * sr  # 5 minutes
            if len(audio) > max_length:
                audio = audio[:max_length]
                logger.warning(
                    f"Audio truncated to {max_length/sr:.1f} seconds")

            # Save processed audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name

            torchaudio.save(temp_path, torch.tensor(audio).unsqueeze(0), sr)
            return temp_path

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            # Return original path if preprocessing fails
            return audio_path

    def _build_audio_grading_prompt(self,
                                    presentation_type: str,
                                    audience: str,
                                    goals: List[str],
                                    custom_goals: str,
                                    transcript: Optional[str] = None) -> str:
        """
        Adapt the existing prompt for audio-based grading using Qwen2-Audio format.
        Keeps the same structure as the original AudioGradingAgent but adds audio analysis.
        """
        goals_text = "; ".join(goals or [])
        if custom_goals:
            goals_text += f"; {custom_goals}"

        rubric_text = json.dumps(self.rubric, indent=2)

        # Use the SAME prompt structure as your existing AudioGradingAgent
        # but with audio-specific instructions
        prompt = f""" <|audio_bos|><|AUDIO|><|audio_eos|>You are an expert speaking coach and grader.
 
The speaker is preparing for a "{presentation_type}" targeting "{audience}". Their specific goals are: {goals_text}.
 
Give feedback and suggestions customized to the type of presentation, audience, and these goals.
Be culturally sensitive: DO NOT suggest changes to accent or cultural speech style. Focus on clarity, pacing, confidence, engagement, and natural delivery. DO NOT mention accent or compare to native speakers.
 
IMPORTANT: You can hear the actual audio recording. Base your analysis on what you HEAR, not just the transcript.
 
1. Rewrite the student's speech to improve grammar, clarity, flow, and natural speech (without changing their intent or topic).
2. Assign an overall score (1-5) for the speech using the rubric and weighing most relevant categories for this context.
3. For each rubric category (see below), give:
   - The category name,
   - The score (1-5),
   - Concise, actionable feedback for that category.
   - Key suggestions for improvement for that category (max 2).
4. Extract the top 4 "Key Moments" from the audio. These should be specific highlights or issues (from any rubric category). For each, provide:
   - "category": The rubric category this moment is most related to,
   - A type ("positive" or "issue")
   - A short message (5-12 words, e.g., "Excellent opening statement" or "Minor mumble on 'statistics'")
   - The timestamp (in seconds, as a float or integer, e.g., 12 or 12.3), using the audio timing for when this moment occurs.
 
Return ONLY this JSON format and nothing else:
{{
  "script": "<string>",
  "score": <number>,
  "overall_feedback": "<string>",
  "rubric_details": [
    {{
      "category": "<rubric category name>",
      "score": <number>,
      "feedback": "<feedback for this category>",
      "suggestions": ["<suggestion1>", "<suggestion2>"]
    }},
    ...
  ],
  "key_moments": [
    {{"category": "<category name>", "type": "<positive/issue>", "message": "<short message>", "timestamp": <seconds>}}
  ]
}}

Here is the rubric (JSON):
{rubric_text}

{f'Here is the transcript for reference: """{transcript}"""' if transcript else 'No transcript provided - analyze the audio directly.'}
"""

        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the Qwen2-Audio response and extract the grading results.

        Args:
            response: Raw response from Qwen2-Audio

        Returns:
            Parsed grading results dictionary
        """
        try:
            # Find JSON in response (Qwen2-Audio might include extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)

            # Validate required fields
            required_fields = ['script', 'score',
                               'overall_feedback', 'rubric_details']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing required field: {field}")

            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Qwen2-Audio response: {e}")
            logger.error(f"Raw response: {response}")

            # Fallback response
            return {
                "script": "Analysis failed - could not parse response",
                "score": 3,
                "overall_feedback": "Unable to analyze audio due to parsing error",
                "rubric_details": [],
                "key_moments": [],
                "audio_analysis": {
                    "error": "Response parsing failed"
                }
            }

    def grade_audio_delivery(self,
                             audio_path: str,
                             presentation_type: str,
                             audience: str,
                             goals: List[str],
                             custom_goals: str,
                             transcript: Optional[str]) -> Dict[str, Any]:
        """
        Grade audio delivery using direct audio analysis with Qwen2-Audio.

        Args:
            audio_path: Path to the audio file to analyze
            presentation_type: Type of presentation
            audience: Target audience
            goals: List of presentation goals
            custom_goals: Additional custom goals
            transcript: Optional transcript for additional context

        Returns:
            Comprehensive grading results dictionary
        """
        try:
            # Ensure model is loaded and update usage tracking
            self._load_model()
            self._update_last_used()

            # Preprocess audio
            processed_audio_path = self._preprocess_audio(audio_path)

            audio, sr = librosa.load(processed_audio_path, sr=16000)
            # Build prompt
            prompt = self._build_audio_grading_prompt(
                presentation_type, audience, goals, custom_goals, transcript
            )

            logger.info("Processing audio with Qwen2-Audio...")

            # Process with Qwen2-Audio
            inputs = self.processor(
                text=prompt,
                audio=audio,
                return_tensors="pt",
                sampling_rate=16000
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode response
            response = self.processor.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # Clean up temporary file
            if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                os.unlink(processed_audio_path)

            # Parse and return results
            result = self._parse_response(response)
            logger.info("✅ Audio grading completed successfully")

            return result

        except Exception as e:
            logger.error(f"❌ Audio grading failed: {e}")

            # Cleanup
            if 'processed_audio_path' in locals() and processed_audio_path != audio_path:
                try:
                    os.unlink(processed_audio_path)
                except:
                    pass

            # Return error response
            return {
                "script": "Analysis failed due to technical error",
                "score": 3,
                "overall_feedback": f"Audio analysis encountered an error: {str(e)}",
                "rubric_details": [],
                "key_moments": [],
                "audio_analysis": {
                    "error": str(e)
                }
            }

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the Qwen2-Audio agent.

        Returns:
            Health status dictionary
        """
        try:
            current_time = time.time()
            time_since_last_use = current_time - self.last_used

            gpu_info = None
            if torch.cuda.is_available():
                gpu_info = {
                    "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "allocated_memory_gb": torch.cuda.memory_allocated() / (1024**3),
                    "cached_memory_gb": torch.cuda.memory_reserved() / (1024**3)
                }

            return {
                "status": "healthy",
                "model_loaded": self.model_loaded,
                "device": self.device,
                "model_name": self.model_name,
                "idle_timeout": self.idle_timeout,
                "time_since_last_use": time_since_last_use,
                "cleanup_scheduled": self.cleanup_timer is not None and self.cleanup_timer.is_alive() if self.cleanup_timer else False,
                "cuda_available": torch.cuda.is_available(),
                "gpu_memory": gpu_info
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model_loaded": self.model_loaded
            }

    def __del__(self):
        """Cleanup when the agent is destroyed."""
        try:
            if hasattr(self, 'model_loaded') and self.model_loaded:
                self.unload_model()
        except Exception as e:
            logger.error(f"Error during agent destruction: {e}")
