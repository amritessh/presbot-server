#!/usr/bin/env python3
"""
Mock version of integratedAssessment.py for testing
"""
import sys
import json

def mock_voicecoach_result():
    """Return a mock VoiceCoach result for testing"""
    return {
        "audio": "test_audio.wav",
        "asr": {
            "language": "en",
            "duration": 15.5,
            "text": "Hello, this is a test speech for presentation grading. I am practicing my delivery and trying to improve my clarity and confidence."
        },
        "metrics": {
            "duration_sec": 15.5,
            "wpm": 120.0,
            "filler_ratio": 0.02,
            "pause_ratio": 0.25,
            "clarity_index": 0.8,
            "tone_variability": 0.7,
            "pacing_score": 0.75
        },
        "rubric": {
            "Clarity": {
                "score": 4,
                "why": "Speech is clear and articulate with good pronunciation"
            },
            "Confidence": {
                "score": 3,
                "why": "Shows some confidence but could be stronger"
            },
            "Tone": {
                "score": 4,
                "why": "Appropriate tone for the audience and content"
            },
            "Pacing": {
                "score": 3,
                "why": "Good pacing overall with room for improvement"
            },
            "Engagement": {
                "score": 4,
                "why": "Engaging delivery that maintains listener interest"
            },
            "Cadence": {
                "score": 3,
                "why": "Natural rhythm with some minor variations"
            },
            "Flow": {
                "score": 4,
                "why": "Smooth flow with good transitions between ideas"
            }
        },
        "feedback": "Good overall presentation with clear articulation and appropriate tone. Focus on building more confidence in your delivery - try practicing with power poses before speaking. Your pacing is solid and you maintain good engagement with the audience. Work on varying your cadence slightly more to add interest to your speech."
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python integratedAssessment_mock.py <audio_file_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    print(f"Processing audio file: {audio_path}")
    print("=== FINAL JSON OUTPUT ===")
    print(json.dumps(mock_voicecoach_result(), indent=2))
