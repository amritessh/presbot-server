import json
import requests
from typing import Dict, List, Any


class AudioGradingAgent:
    """
    Agent responsible for grading audio/speech delivery using the speech rubric.
    This agent evaluates HOW the person speaks (clarity, confidence, tone, pacing, etc.)
    """

    def __init__(self, llama_url: str = "http://localhost:5000/api/chat",
                 llama_model: str = "llama3.2-vision:11b",
                 rubric_path: str = "config/rubric.json"):
        self.llama_url = llama_url
        self.llama_model = llama_model
        self.rubric_path = rubric_path

        # Load audio/speech rubric
        with open(self.rubric_path, 'r') as f:
            self.rubric = json.load(f)

    def grade_audio_delivery(self, transcript: str, presentation_type: str,
                             audience: str, goals: List[str], custom_goals: str) -> Dict[str, Any]:
        """
        Grade audio/speech delivery and generate refined script.
        Uses the exact same logic as the existing grade_audio_with_llama function.
        """
        rubric_text = json.dumps(self.rubric, indent=2)
        goals_text = "; ".join(goals or [])
        if custom_goals:
            goals_text += f"; {custom_goals}"

        prompt = f"""You are an expert speaking coach and grader.
 
The speaker is preparing for a "{presentation_type}" targeting "{audience}". Their specific goals are: {goals_text}.
 
Give feedback and suggestions customized to the type of presentation, audience, and these goals.
Be culturally sensitive: DO NOT suggest changes to accent or cultural speech style. Focus on clarity, pacing, confidence, engagement, and natural delivery. DO NOT mention accent or compare to native speakers.
 
1. Rewrite the student's speech to improve grammar, clarity, flow, and natural speech (without changing their intent or topic).
2. Assign an overall score (1-5) for the speech using the rubric and weighing most relevant categories for this context.
3. For each rubric category (see below), give:
    - The category name,
    - The score (1-5),
    - Concise, actionable feedback for that category.
    - Key suggestions for improvement for that category (max 2).
4. Extract the top 4 "Key Moments" from the transcript. These should be specific highlights or issues (from any rubric category). For each, provide:
    - "category": The rubric category this moment is most related to,
    - A type ("positive" or "issue")
    - A short message (5-12 words, e.g., "Excellent opening statement" or "Minor mumble on 'statistics'")
    - The timestamp (in seconds, as a float or integer, e.g., 12 or 12.3), using the transcript timing for when this moment occurs.
 
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
 
Here is the transcript to be graded (with timestamps):
\"\"\"
{transcript}
\"\"\"
"""


        try:
            llama_response = requests.post(
                self.llama_url,
                json={
                    "model": self.llama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=600  # 10 minute timeout for large files
            )
            llama_response.raise_for_status()

            result = llama_response.json()
            llama_output = result['message']['content']
            print("RAW LLAMA OUTPUT:", llama_output)

            try:
                output_json = json.loads(llama_output)
                return output_json
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse JSON from Llama output. Raw: {llama_output}")

        except Exception as e:
            raise Exception(f"Audio grading failed: {str(e)}")
