"""
AI Advisor Module
=================
Uses Google Gemini to provide intelligent guidance for Newton-Raphson solving.
"""

from google import genai
import os
from typing import Dict, List, Optional


class GeminiAdvisor:
    """AI advisor using Google Gemini for Newton-Raphson guidance."""
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini with API key."""
        if api_key:
            self.api_key = api_key
        else:
            # Support comma-separated key list from AI_GEN_API_KEYS, or single GOOGLE_API_KEY
            keys_str = os.getenv('AI_GEN_API_KEYS', '')
            if keys_str:
                # Strip quotes and pick the first key
                keys_str = keys_str.strip('"').strip("'")
                self.api_key = keys_str.split(',')[0].strip()
            else:
                self.api_key = os.getenv('GOOGLE_API_KEY', '')
        
        if not self.api_key:
            self.available = False
            return
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = 'gemini-2.5-flash'
            self.available = True
        except Exception:
            self.available = False
    
    def suggest_initial_guesses(self, function_str: str) -> Dict:
        """Suggest good starting points for Newton-Raphson."""
        if not self.available:
            return self._fallback_suggestions()
        
        prompt = f"""You are a numerical analysis expert. For f(x) = {function_str},
suggest 3 good initial guesses for Newton-Raphson method.

Consider sign changes, derivative magnitude, and avoid singularities.

Return ONLY valid JSON (no markdown, no code blocks):
{{"guesses": [val1, val2, val3], "reasoning": "brief explanation"}}"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return self._parse_json(response.text)
        except Exception:
            return self._fallback_suggestions()
    
    def diagnose_failure(self, function_str: str, history: List[Dict], 
                          issues: List[str]) -> str:
        """Diagnose why Newton-Raphson failed."""
        if not self.available:
            return self._fallback_diagnosis(issues)
        
        history_str = self._format_history(history)
        
        prompt = f"""Numerical analysis expert diagnosing Newton-Raphson failure.

Function: f(x) = {function_str}
Issues: {issues}
Iterations:
{history_str}

Provide in 3 short sections:
1. ROOT CAUSE: Why it failed (2 sentences)
2. FIX: Specific suggestion with numbers (1-2 sentences)  
3. ALTERNATIVE: If Newton won't work, what method? (1 sentence)"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception:
            return self._fallback_diagnosis(issues)
    
    def explain_step(self, function_str: str, step: Dict, 
                      prev_step: Optional[Dict] = None) -> str:
        """Explain what happened at a specific iteration."""
        if not self.available:
            return f"At x = {step['x']:.6f}, f(x) = {step['f_x']:.6f}. " \
                   f"The {'error is decreasing' if prev_step and abs(step['f_x']) < abs(prev_step['f_x']) else 'method is progressing'}."
        
        prev_info = ""
        if prev_step:
            prev_info = f"Previous: x = {prev_step['x']:.6f}, f(x) = {prev_step['f_x']:.6f}\n"
        
        prompt = f"""Explain this Newton-Raphson step in 2 sentences:

f(x) = {function_str}
{prev_info}
Current (iteration {step['iteration']}):
- x = {step['x']:.6f}
- f(x) = {step['f_x']:.6f}
- f'(x) = {step['df_x']:.6f}
- Step taken = {step['step_size']:.6f}

What does this tell us about convergence?"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception:
            return f"Step {step['iteration']}: Moved to x = {step['x']:.6f}"
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format history for prompt."""
        if len(history) <= 5:
            lines = [f"  Iter {h['iteration']}: x={h['x']:.4f}, f(x)={h['f_x']:.4f}" 
                    for h in history]
        else:
            lines = [f"  Iter {h['iteration']}: x={h['x']:.4f}, f(x)={h['f_x']:.4f}" 
                    for h in history[:3]]
            lines.append("  ...")
            lines.extend([f"  Iter {h['iteration']}: x={h['x']:.4f}, f(x)={h['f_x']:.4f}" 
                         for h in history[-2:]])
        return "\n".join(lines)
    
    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from Gemini response."""
        import json
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except Exception:
            return self._fallback_suggestions()
    
    def _fallback_suggestions(self) -> Dict:
        """Default suggestions when AI unavailable."""
        return {
            "guesses": [-2.0, 0.0, 2.0],
            "reasoning": "Default guesses (AI unavailable). Try points where function might cross zero."
        }
    
    def _fallback_diagnosis(self, issues: List[str]) -> str:
        """Default diagnosis when AI unavailable."""
        messages = {
            "ZERO_DERIVATIVE": "The derivative became zero, causing division by zero. Try a different starting point away from critical points.",
            "OSCILLATING": "The method is bouncing between values. Try a starting point closer to the root or use bisection method.",
            "DIVERGING": "The iterations are moving away from the root. The initial guess may be too far. Try a smaller starting value.",
            "MAX_ITERATIONS_REACHED": "Too many iterations needed. Increase max iterations or check if a root exists.",
            "NUMERICAL_INSTABILITY": "Numbers became too large or NaN. Check for singularities in the function.",
            "SMALL_DERIVATIVE_AT_START": "Starting near a flat region. Move initial guess to where slope is steeper."
        }
        diagnosis = "ROOT CAUSE: " + ". ".join(messages.get(i, i) for i in issues)
        diagnosis += "\n\nFIX: Try different initial guesses, especially where f(x) changes sign."
        diagnosis += "\n\nALTERNATIVE: Consider bisection method if Newton-Raphson continues to fail."
        return diagnosis