import json
import requests
import re
from typing import Dict, List, Optional, Any, Tuple


def load_script_rubric() -> Dict[str, Any]:
    """Load the script-specific rubric for evaluating written content."""
    with open('config/script_rubric.json', 'r') as f:
        return json.load(f)


def get_context_weights(rubric: Dict[str, Any], presentation_type: str) -> Dict[str, float]:
    """Get context-specific weights for categories based on presentation type."""
    weighting_config = rubric.get("weighting_by_context", {})
    contexts = weighting_config.get("contexts", {})

    # Normalize presentation type for lookup
    normalized_type = presentation_type.lower().replace(" ", "_")

    # Try exact match first, then partial matches
    if normalized_type in contexts:
        return contexts[normalized_type]

    # Fallback mapping for common variations
    type_mapping = {
        "class": "class_presentation",
        "business": "business_presentation",
        "sales": "sales_pitch",
        "academic": "academic_conference",
        "public": "public_speaking"
    }

    for key, mapped_type in type_mapping.items():
        if key in normalized_type and mapped_type in contexts:
            return contexts[mapped_type]

    # Default to equal weights if no match found
    categories = [k for k in rubric.keys() if k != "weighting_by_context"]
    return {cat: 1.0/len(categories) for cat in categories}


class ComparativeGradingAgent:
    """
    Agent that grades both original and refined scripts to validate improvements
    and provide detailed before/after analysis using script-specific criteria.
    """

    def __init__(self, llama_url: str = "http://localhost:5000/api/chat",
                 llama_model: str = "llama3.2-vision:11b"):
        self.llama_url = llama_url
        self.llama_model = llama_model
        self.rubric = load_script_rubric()

    def grade_single_script(self, script: str, presentation_type: str,
                            audience: str, goals: List[str], custom_goals: str,
                            script_label: str = "Script") -> Dict[str, Any]:
        """Grade a single script using the script-specific rubric."""

        context_weights = get_context_weights(self.rubric, presentation_type)

        goals_text = "; ".join(goals or [])
        if custom_goals:
            goals_text += f"; {custom_goals}"

        prompt = f"""You are an expert writing coach evaluating script content. Return ONLY valid JSON, no other text.

CONTEXT:
- Presentation Type: "{presentation_type}"
- Target Audience: "{audience}"
- Speaker Goals: {goals_text}
- Script Type: {script_label}

INSTRUCTIONS:
1. Evaluate this WRITTEN SCRIPT for content quality and delivery readiness
2. Score each category 1-5 based on the written content only
3. Focus on what can be measured in text: grammar, structure, tone, engagement potential
4. Return ONLY the JSON below - no explanations, no markdown, no other text

SCRIPT RUBRIC CATEGORIES:
{json.dumps({k: v for k, v in self.rubric.items() if k != "weighting_by_context"}, indent=2)}

RETURN ONLY THIS JSON FORMAT:
{{
  "script_type": "{script_label}",
  "category_scores": {{
    "Content Quality": <1-5>,
    "Structure & Organization": <1-5>,
    "Professionalism & Tone": <1-5>, 
    "Engagement Potential": <1-5>,
    "Delivery Readiness": <1-5>
  }},
  "overall_score": <weighted average>,
  "rubric_details": [
    {{
      "category": "Content Quality",
      "score": <1-5>,
      "feedback": "<brief analysis of grammar, word choice, clarity>",
      "key_observations": ["<observation 1>", "<observation 2>"]
    }},
    {{
      "category": "Structure & Organization",
      "score": <1-5>, 
      "feedback": "<brief analysis of logical flow and organization>",
      "key_observations": ["<observation 1>", "<observation 2>"]
    }},
    {{
      "category": "Professionalism & Tone",
      "score": <1-5>,
      "feedback": "<brief analysis of appropriateness for audience>", 
      "key_observations": ["<observation 1>", "<observation 2>"]
    }},
    {{
      "category": "Engagement Potential",
      "score": <1-5>,
      "feedback": "<brief analysis of how engaging content would be>",
      "key_observations": ["<observation 1>", "<observation 2>"]
    }},
    {{
      "category": "Delivery Readiness", 
      "score": <1-5>,
      "feedback": "<brief analysis of readability and speakability>",
      "key_observations": ["<observation 1>", "<observation 2>"]
    }}
  ]
}}

SCRIPT TO EVALUATE:
\"\"\"
{script}
\"\"\"
"""

        try:
            response = requests.post(
                self.llama_url,
                json={
                    "model": self.llama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json"
                },
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            llama_output = result['message']['content']

            # Parse JSON response
            try:
                output_json = json.loads(llama_output)

                # Calculate weighted score
                category_scores = output_json.get('category_scores', {})
                if category_scores:
                    total_weighted_score = 0.0
                    total_weight = 0.0
                    for category, score in category_scores.items():
                        weight = context_weights.get(category, 0.2)
                        total_weighted_score += score * weight
                        total_weight += weight

                    output_json['overall_score'] = round(
                        total_weighted_score / total_weight, 1)
                    output_json['context_weights_used'] = context_weights

                return output_json

            except json.JSONDecodeError:
                # Try extracting from code blocks
                json_match = re.search(
                    r'```(?:json)?\s*(\{.*?\})\s*```', llama_output, re.DOTALL)
                if json_match:
                    output_json = json.loads(json_match.group(1))
                    return output_json

                raise ValueError(
                    f"Could not parse JSON from LLM output: {llama_output}")

        except Exception as e:
            raise RuntimeError(f"Failed to grade {script_label}: {e}")

    def compare_scripts(self, original_script: str, refined_script: str,
                        presentation_type: str, audience: str,
                        goals: List[str], custom_goals: str) -> Dict[str, Any]:
        """
        Compare original and refined scripts with detailed analysis of improvements.
        """

        # Grade both scripts
        print("Grading original script...")
        original_results = self.grade_single_script(
            original_script, presentation_type, audience, goals, custom_goals, "Original Script"
        )

        print("Grading refined script...")
        refined_results = self.grade_single_script(
            refined_script, presentation_type, audience, goals, custom_goals, "Refined Script"
        )

        # Generate detailed comparison
        print("Generating comparison analysis...")
        comparison_analysis = self._generate_comparison_analysis(
            original_results, refined_results, original_script, refined_script,
            presentation_type, audience, goals, custom_goals
        )

        return {
            "original_results": original_results,
            "refined_results": refined_results,
            "comparison_analysis": comparison_analysis,
            "summary": self._create_summary(original_results, refined_results)
        }

    def _generate_comparison_analysis(self, original_results: Dict, refined_results: Dict,
                                      original_script: str, refined_script: str,
                                      presentation_type: str, audience: str,
                                      goals: List[str], custom_goals: str) -> Dict[str, Any]:
        """Generate detailed comparison analysis using LLM."""

        goals_text = "; ".join(goals or [])
        if custom_goals:
            goals_text += f"; {custom_goals}"

        prompt = f"""You are an expert writing coach comparing two written scripts for the same presentation. Return ONLY valid JSON.

CONTEXT:
- Presentation Type: "{presentation_type}"
- Audience: "{audience}"
- Speaker Goals: {goals_text}

SCRIPT SCORES:
- Original: {original_results.get('overall_score', 0)}/5
- Refined: {refined_results.get('overall_score', 0)}/5

CATEGORY SCORE CHANGES:
{self._format_score_changes(original_results, refined_results)}

TASK:
1. Identify key content and writing changes between the scripts.
2. Evaluate effectiveness of grammar, structure, tone, and delivery improvements.
3. Highlight changes with the strongest impact on readability and oral delivery.
4. Focus on written quality and preparedness for speaking.
5. Suggest further refinements for the script or for spoken delivery.

RETURN ONLY THIS JSON:
{{
  "improvement_score": <refined_score - original_score>,
  "improvement_percentage": <percentage improvement>,
  "effectiveness_rating": "<Excellent/Good/Moderate/Minimal/Poor>",
  "key_changes_made": [
    {{
      "change_type": "<Content/Structure/Tone/Engagement/Delivery>",
      "description": "<summary of the change>",
      "impact": "<High/Medium/Low>",
      "example": "<specific difference between scripts>"
    }}
  ],
  "most_impactful_improvements": [
    "<writing improvement 1>",
    "<writing improvement 2>", 
    "<writing improvement 3>"
  ],
  "potential_concerns": [
    "<concern 1 (if any)>",
    "<concern 2 (if any)>"
  ],
  "additional_suggestions": [
    "<refinement suggestion 1>",
    "<refinement suggestion 2>"
  ],
  "validation_summary": "<2–3 sentence assessment of whether the refinements improved the script>"
}}

ORIGINAL SCRIPT:
\"\"\"
{original_script}
\"\"\"

REFINED SCRIPT:
\"\"\"
{refined_script}
\"\"\"
"""

        try:
            response = requests.post(
                self.llama_url,
                json={
                    "model": self.llama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json"
                },
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            llama_output = result['message']['content']

            try:
                return json.loads(llama_output)
            except json.JSONDecodeError:
                json_match = re.search(
                    r'```(?:json)?\s*(\{.*?\})\s*```', llama_output, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                raise ValueError(
                    f"Could not parse comparison analysis: {llama_output}")

        except Exception as e:
            return {"error": f"Failed to generate comparison analysis: {e}"}

    def _format_score_changes(self, original: Dict, refined: Dict) -> str:
        """Format score changes for display in prompt."""
        original_scores = original.get('category_scores', {})
        refined_scores = refined.get('category_scores', {})

        changes = []
        for category in original_scores.keys():
            orig_score = original_scores.get(category, 0)
            refined_score = refined_scores.get(category, 0)
            change = refined_score - orig_score
            change_str = f"+{change}" if change > 0 else str(change)
            changes.append(
                f"{category}: {orig_score} → {refined_score} ({change_str})")

        return "\n".join(changes)

    def _create_summary(self, original: Dict, refined: Dict) -> Dict[str, Any]:
        """Create a concise summary of the comparison."""
        original_score = original.get('overall_score', 0)
        refined_score = refined.get('overall_score', 0)
        improvement = refined_score - original_score
        improvement_pct = (improvement / original_score *
                           100) if original_score > 0 else 0

        # Count category improvements
        original_scores = original.get('category_scores', {})
        refined_scores = refined.get('category_scores', {})

        categories_improved = 0
        categories_declined = 0
        categories_same = 0

        for category in original_scores.keys():
            orig = original_scores.get(category, 0)
            refined = refined_scores.get(category, 0)

            if refined > orig:
                categories_improved += 1
            elif refined < orig:
                categories_declined += 1
            else:
                categories_same += 1

        return {
            "original_score": original_score,
            "refined_score": refined_score,
            "improvement": round(improvement, 1),
            "improvement_percentage": round(improvement_pct, 1),
            "categories_improved": categories_improved,
            "categories_declined": categories_declined,
            "categories_unchanged": categories_same,
            "overall_assessment": self._get_improvement_assessment(improvement)
        }

    def _get_improvement_assessment(self, improvement: float) -> str:
        """Get qualitative assessment of improvement."""
        if improvement >= 1.0:
            return "Significant Improvement"
        elif improvement >= 0.5:
            return "Good Improvement"
        elif improvement >= 0.2:
            return "Moderate Improvement"
        elif improvement > 0:
            return "Slight Improvement"
        elif improvement == 0:
            return "No Change"
        else:
            return "Declined"


def run_comparative_analysis(original_transcript: str, refined_script: str,
                             presentation_type: str = "Class Presentation",
                             audience: str = "Students",
                             goals: List[str] = None,
                             custom_goals: str = "") -> Dict[str, Any]:
    """
    Convenience function to run comparative analysis.

    Args:
        original_transcript: The original speech transcript
        refined_script: The improved/refined script
        presentation_type: Type of presentation
        audience: Target audience
        goals: List of presentation goals
        custom_goals: Additional custom goals

    Returns:
        Complete comparative analysis results
    """

    agent = ComparativeGradingAgent()

    try:
        results = agent.compare_scripts(
            original_script=original_transcript,
            refined_script=refined_script,
            presentation_type=presentation_type,
            audience=audience,
            goals=goals or [],
            custom_goals=custom_goals
        )

        return results

    except Exception as e:
        return {"error": f"Comparative analysis failed: {e}"}


def print_comparison_summary(results: Dict[str, Any]):
    """Print a formatted summary of comparison results."""

    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return

    summary = results.get("summary", {})
    comparison = results.get("comparison_analysis", {})

    print("=" * 60)
    print("📊 SCRIPT COMPARATIVE ANALYSIS")
    print("=" * 60)

    print(f"\n📈 SCRIPT IMPROVEMENT SCORES:")
    print(f"   Original: {summary.get('original_score', 0)}/5")
    print(f"   Refined:  {summary.get('refined_score', 0)}/5")
    print(
        f"   Change:   {summary.get('improvement', 0):+.1f} ({summary.get('improvement_percentage', 0):+.1f}%)")
    print(f"   Assessment: {summary.get('overall_assessment', 'Unknown')}")

    print(f"\n📊 CATEGORY CHANGES:")
    print(f"   ✅ Improved: {summary.get('categories_improved', 0)}")
    print(f"   ❌ Declined: {summary.get('categories_declined', 0)}")
    print(f"   ➖ Unchanged: {summary.get('categories_unchanged', 0)}")

    if comparison.get('most_impactful_improvements'):
        print(f"\n🎯 MOST IMPACTFUL WRITING IMPROVEMENTS:")
        for i, improvement in enumerate(comparison['most_impactful_improvements'], 1):
            print(f"   {i}. {improvement}")

    if comparison.get('potential_concerns'):
        print(f"\n⚠️  POTENTIAL CONCERNS:")
        for concern in comparison['potential_concerns']:
            print(f"   • {concern}")

    print(
        f"\n💡 VALIDATION: {comparison.get('validation_summary', 'No summary available')}")
