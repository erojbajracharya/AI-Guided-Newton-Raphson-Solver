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
            msg = f"**Newton-Raphson Iteration {step['iteration']} (Fallback explanation)**\n\n"
            msg += f"- **Current Guess (x):** `{step['x']:.6f}`\n"
            msg += f"- **Function value f(x):** `{step['f_x']:.2e}` (this is our vertical height; we want this to be zero)\n"
            msg += f"- **Slope/Derivative f'(x):** `{step['df_x']:.6f}` (the direction/steepness at our current guess)\n"
            msg += f"- **Step size taken:** `{step['step_size']:.2e}` (distance we moved along the x-axis to get here)\n\n"
            
            if prev_step:
                curr_err = abs(step['f_x'])
                prev_err = abs(prev_step['f_x'])
                if curr_err < prev_err:
                    improvement = (prev_err - curr_err) / prev_err * 100 if prev_err != 0 else 0
                    msg += f"📊 **Progress:** The error decreased by {improvement:.1f}%. The solver is successfully moving closer to the root!"
                else:
                    msg += "⚠️ **Warning:** The error increased compared to the previous step. This means we might be oscillating or moving away from the root."
            else:
                msg += "🌱 **Start:** This is the initial step based on the starting guess $x_0$."
            return msg
        
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
            "guesses": [-2.0, 0.5, 2.0],
            "reasoning": "Fallback Mode (AI key not set): Try evaluating the function at these standard test points to find where it changes sign, indicating a root is nearby."
        }
    
    def _fallback_diagnosis(self, issues: List[str]) -> str:
        """Default diagnosis when AI unavailable."""
        messages = {
            "ZERO_DERIVATIVE": "The derivative became zero. Since the Newton-Raphson formula divides by f'(x), a zero slope results in division by zero (tangent is parallel to the x-axis).",
            "ZERO_OR_SMALL_DERIVATIVE": "The derivative is zero or extremely close to zero. This causes division by zero or an extremely large step size, pushing the next guess far away from the root.",
            "OSCILLATING": "The solver is bouncing back and forth between two or more values (infinitely looping). This happens when the initial guess is caught in a cycle near local minima or maxima.",
            "DIVERGING": "The iterations are moving further away from the root rather than approaching it. The starting point might be too far from the root or on a steep slope facing away.",
            "MAX_ITERATIONS_REACHED": "The solver reached the maximum iterations allowed without satisfying the convergence criteria. The function might not have a real root, or the tolerance is set too strict.",
            "NUMERICAL_INSTABILITY": "The calculation resulted in numbers that are too large (overflow), too small (underflow), or not a number (NaN). This occurs when evaluating undefined points or division by zero.",
            "SMALL_DERIVATIVE_AT_START": "The starting point has a very flat slope, meaning the initial tangent line points far away from the root. It is better to start at a point with a larger slope."
        }
        diagnosis = "ROOT CAUSE: " + ". ".join(messages.get(i, i) for i in issues)
        diagnosis += "\n\nFIX: Try different initial guesses, especially where f(x) changes sign to bracket a root."
        diagnosis += "\n\nALTERNATIVE: Consider the Bisection or Secant method if Newton-Raphson continues to fail."
        return diagnosis