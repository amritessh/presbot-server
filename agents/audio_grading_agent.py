"""
Voice-based audio grading agent that uses VoiceClone technical speech analysis.
"""

import json
import os
import subprocess
import re
import requests
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VoiceGradingAgent:
    """
    Agent that uses VoiceClone technical analysis for speech assessment.
    """

    def __init__(self, 
                 llama_url: str = "http://localhost:5000/api/generate",
                 llama_model: str = "gemma3:12b"):
        """
        Initialize the VoiceClone grading agent.
        """
        self.llama_url = llama_url
        self.llama_model = llama_model

    def run_voiceclone_analysis(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Run VoiceClone analysis using integratedAssessment.py
        """
        try:
            logger.info("🔬 Running VoiceClone technical analysis...")
            
            # Run VoiceClone analysis via integratedAssessment.py
            result = subprocess.run(
                ["python3", "integratedAssessment.py", audio_path],
                capture_output=True,
                text=True,
                check=True,
                cwd="/Users/patel.hetas/Projects/presbot-server"  # Run from project root
            )

            output = result.stdout.strip()
            
            # Parse JSON output 
            try:
                # Look for JSON after "=== FINAL JSON OUTPUT ===" marker
                if "=== FINAL JSON OUTPUT ===" in output:
                    json_start = output.find("=== FINAL JSON OUTPUT ===")
                    json_part = output[json_start + len("=== FINAL JSON OUTPUT ==="):].strip()
                    data = json.loads(json_part)
                    logger.info("✅ VoiceClone analysis completed successfully")
                    return data
                else:
                    # Fallback: Look for the largest JSON block
                    json_blocks = re.findall(r'\{.*?\}', output, re.DOTALL)
                    if json_blocks:
                        largest_json = max(json_blocks, key=len)
                        data = json.loads(largest_json)
                        logger.info("✅ VoiceClone analysis completed successfully")
                        return data
                    else:
                        logger.error(f"No JSON found in VoiceClone output: {output}")
                        return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Could not parse JSON from VoiceClone output: {e}")
                logger.error(f"Raw output: {output}")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ VoiceClone analysis failed: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error in VoiceClone analysis: {e}")
            return None

    def _parse_feedback_sections(self, raw_feedback: str) -> Dict[str, Any]:
        """
        Parse VoiceClone feedback into organized sections.
        """
        if not raw_feedback:
            return {
                "summary": "No feedback available",
                "priority_areas": [],
                "strengths": []
            }
        
        try:
            # Split feedback into sections based on common patterns
            sections = {
                "summary": "",
                "priority_areas": [],
                "strengths": []
            }
            
            # Extract summary (first paragraph before bullet points)
            lines = raw_feedback.split('\n')
            summary_lines = []
            priority_areas = []
            strengths = []
            
            current_section = "summary"
            current_item = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for priority items (numbered or bullet points)
                if line.startswith('*') and ('Priority' in line or 'Try:' in line):
                    if current_item:
                        priority_areas.append(current_item.strip())
                    current_item = line
                    current_section = "priority"
                elif line.startswith('*') and current_section == "priority":
                    current_item += " " + line
                elif 'strong' in line.lower() or 'excellent' in line.lower() or 'good' in line.lower():
                    if current_item and current_section == "priority":
                        priority_areas.append(current_item.strip())
                        current_item = ""
                    strengths.append(line)
                    current_section = "strengths"
                elif current_section == "summary" and not line.startswith('*'):
                    summary_lines.append(line)
                elif current_section == "priority":
                    current_item += " " + line
            
            # Add final priority item if exists
            if current_item:
                priority_areas.append(current_item.strip())
            
            # Clean up sections
            sections["summary"] = " ".join(summary_lines).strip()
            
            # Clean and format priority areas
            cleaned_priorities = []
            for item in priority_areas:
                # Remove asterisks and clean up
                clean_item = item.replace('*', '').strip()
                if clean_item:
                    cleaned_priorities.append(clean_item)
            sections["priority_areas"] = cleaned_priorities
            
            # Clean strengths
            cleaned_strengths = []
            for item in strengths:
                clean_item = item.replace('*', '').strip()
                if clean_item:
                    cleaned_strengths.append(clean_item)
            sections["strengths"] = cleaned_strengths
            
            # If no sections found, put everything in summary
            if not sections["priority_areas"] and not sections["strengths"]:
                sections["summary"] = raw_feedback
            
            return sections
            
        except Exception as e:
            logger.error(f"Failed to parse feedback sections: {e}")
            return {
                "summary": raw_feedback,
                "priority_areas": [],
                "strengths": []
            }

    def generate_refined_transcript(self, 
                                   original_transcript: str,
                                   voiceclone_data: Dict[str, Any],
                                   presentation_type: str,
                                   audience: str,
                                   goals: List[str],
                                   custom_goals: str) -> str:
        """
        Use LLM to generate refined transcript based on VoiceClone analysis.
        """
        try:
            goals_text = "; ".join(goals or [])
            if custom_goals:
                goals_text += f"; {custom_goals}"

            # Get key metrics for context
            metrics = voiceclone_data.get("metrics", {})
            
            # Build prompt for transcript refinement
            prompt = f"""You are an expert speech coach. Based on technical speech analysis, improve this transcript for better delivery.

**Context:**
- Presentation Type: {presentation_type}
- Audience: {audience}
- Goals: {goals_text}

**Technical Analysis Summary:**
- Words per minute: {metrics.get('wpm', 'N/A')}
- Clarity index: {metrics.get('clarity_index', 'N/A')}
- Filler ratio: {metrics.get('filler_ratio', 'N/A')}
- Pause ratio: {metrics.get('pause_ratio', 'N/A')}

**Original Transcript:**
{original_transcript}

**Task:** Rewrite the transcript to improve:
1. Grammar and sentence structure
2. Clarity and flow
3. Natural speech patterns
4. Appropriate tone for the audience and presentation type

Keep the same meaning and main points but make it more suitable for effective delivery.

Return ONLY the improved transcript text (no explanations, no JSON):"""

            # Call LLM
            response = requests.post(
                self.llama_url,
                json={
                    "model": self.llama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            refined_transcript = result.get('response', '').strip()
            
            # Clean up any extra formatting
            refined_transcript = re.sub(r'^```.*?```$', '', refined_transcript, flags=re.DOTALL)
            refined_transcript = refined_transcript.strip('"\'`')
            
            if refined_transcript:
                logger.info("✅ Generated refined transcript")
                return refined_transcript
            else:
                logger.warning("Empty response from LLM, using original transcript")
                return original_transcript

        except Exception as e:
            logger.error(f"❌ Failed to generate refined transcript: {e}")
            return original_transcript  # Fallback to original

    def extract_useful_data(self, voiceclone_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract just the essential data: score, feedback, and key metrics.
        """
        # Extract rubric scores and calculate overall score
        rubric = voiceclone_results.get("rubric", {})
        scores = [data.get("score", 3) for data in rubric.values()]
        overall_score = round(sum(scores) / len(scores), 1) if scores else 3.0
        
        # Extract and parse feedback into sections
        raw_feedback = voiceclone_results.get("feedback", "")
        parsed_feedback = self._parse_feedback_sections(raw_feedback)
        
        # Extract key technical metrics
        metrics = voiceclone_results.get("metrics", {})
        key_metrics = {
            "wpm": metrics.get("wpm", 0),
            "filler_ratio": metrics.get("filler_ratio", 0),
            "pause_ratio": metrics.get("pause_ratio", 0),
            "clarity_index": metrics.get("clarity_index", 0)
        }
        
        return {
            "overall_score": overall_score,
            "feedback": parsed_feedback,
            "key_metrics": key_metrics
        }

    def grade_audio_delivery(self,
                             audio_path: str,
                             presentation_type: str,
                             audience: str,
                             goals: List[str],
                             custom_goals: str,
                             transcript: str) -> Dict[str, Any]:
        """
        Main grading method that gets VoiceClone analysis and extracts useful data.
        """
        try:
            logger.info("🎤 Starting VoiceClone-based audio grading...")
            
            # Step 1: Run VoiceClone analysis
            voiceclone_results = self.run_voiceclone_analysis(audio_path)
            
            if not voiceclone_results:
                return {
                "error": "VoiceClone analysis failed",
                "script": transcript,
                "score": 0,
                "overall_feedback": "Audio analysis could not be completed. Please try again.",
                "key_metrics": {}
                }
            
            # Step 2: Extract useful data from VoiceClone results
            extracted_data = self.extract_useful_data(voiceclone_results)
            
            # Step 3: Generate refined transcript using LLM
            refined_transcript = self.generate_refined_transcript(
                transcript, voiceclone_results, presentation_type, audience, goals, custom_goals
            )
            
            logger.info("✅ VoiceClone grading completed successfully")
            logger.info(f"Extracted score: {extracted_data['overall_score']}")
            
            return {
                "script": refined_transcript,
                "score": extracted_data["overall_score"],
                "overall_feedback": extracted_data["feedback"],
                "key_metrics": extracted_data["key_metrics"],
                # Keep raw output for debugging
                "voiceclone_raw_output": voiceclone_results
            }

        except Exception as e:
            logger.error(f"❌ VoiceClone grading failed: {e}")
            return {
                "error": str(e),
                "script": transcript,
                "score": 0,
                "overall_feedback": f"Audio analysis encountered an error: {str(e)}",
                "key_metrics": {}
            }