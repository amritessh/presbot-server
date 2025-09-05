import json
from typing import Dict, List, Any
from agents.comparative_grading_agent import ComparativeGradingAgent


class ScriptGradingAgent:
    """
    Agent responsible for grading script content using the script rubric.
    This agent evaluates WHAT is written (content quality, structure, professionalism, etc.)
    and provides comparative analysis between original and refined scripts.
    """

    def __init__(self, llama_url: str = "http://localhost:5000/api/chat",
                 llama_model: str = "llama3.2-vision:11b"):
        self.comparative_agent = ComparativeGradingAgent(
            llama_url, llama_model)

    def grade_script_content(self, original_script: str, refined_script: str,
                             presentation_type: str, audience: str,
                             goals: List[str], custom_goals: str) -> Dict[str, Any]:
        """
        Grade script content and provide comparative analysis.
        Uses the existing ComparativeGradingAgent to compare original vs refined scripts.
        """
        try:
            # Use the existing comparative grading functionality
            comparison_results = self.comparative_agent.compare_scripts(
                original_script=original_script,
                refined_script=refined_script,
                presentation_type=presentation_type,
                audience=audience,
                goals=goals,
                custom_goals=custom_goals
            )

            return comparison_results

        except Exception as e:
            raise Exception(f"Script grading failed: {str(e)}")
