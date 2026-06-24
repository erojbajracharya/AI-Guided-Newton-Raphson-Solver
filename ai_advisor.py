"""
AI Advisor Module
=================
Uses Google Gemini to provide intelligent guidance for Newton-Raphson solving.
Supports multiple API keys with automatic rotation.
"""

from typing import Dict, List, Optional
from api_rotator import APIRotator


class GeminiAdvisor:
    """AI advisor using Google Gemini for Newton-Raphson guidance."""
    
    def __init__(self, api_key: str = None):
        """Initialize GeminiAdvisor using the APIRotator."""
        self.rotator = APIRotator()
        if api_key:
            self.rotator.api_keys = [api_key.strip()]
            self.rotator.available = True
        self.available = self.rotator.available
        
    def _try_generate_content(self, prompt: str) -> str:
        """Try each API key one by one until one works."""
        try:
            return self.rotator.generate_content(prompt)
        except Exception as e:
            self.available = self.rotator.available
            raise e
    
    def suggest_initial_guesses(self, function_str: str) -> Dict:
        """Suggest good starting points for Newton-Raphson."""
        if not self.available:
            return {
                "guesses": [],
                "reasoning": "AI Advisor is not connected. Add GOOGLE_API_KEY or AI_GEN_API_KEYS in the .env file to enable AI suggestions."
            }
        
        prompt = f"""You are a numerical analysis expert. For f(x) = {function_str},
suggest 3 good initial guesses for Newton-Raphson method.

Consider sign changes, derivative magnitude, and avoid singularities.

Return ONLY valid JSON (no markdown, no code blocks):
{{"guesses": [val1, val2, val3], "reasoning": "brief explanation"}}"""
        
        try:
            text = self._try_generate_content(prompt)
            return self._parse_json(text)
        except Exception:
            return {
                "guesses": [],
                "reasoning": "AI Advisor could not connect with the available API keys. Please check the keys or try again later."
            }
    
    def diagnose_failure(self, function_str: str, history: List[Dict], 
                          issues: List[str]) -> str:
        """Diagnose why Newton-Raphson failed."""
        if not self.available:
            return "AI Advisor is not connected. Add GOOGLE_API_KEY or AI_GEN_API_KEYS in the .env file to enable failure diagnosis."
        
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
            return self._try_generate_content(prompt)
        except Exception:
            return "AI Advisor could not connect with the available API keys. Please check the keys or try again later."
    
    def explain_step(self, function_str: str, step: Dict, 
                      prev_step: Optional[Dict] = None) -> str:
        """Explain what happened at a specific iteration."""
        if not self.available:
            return "AI Advisor is not connected. Add GOOGLE_API_KEY or AI_GEN_API_KEYS in the .env file to enable step explanation."
        
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
            return self._try_generate_content(prompt)
        except Exception:
            return "AI Advisor could not connect with the available API keys. Please check the keys or try again later."
    
    def answer_question(self, function_str: str, question: str,
                         history: List[Dict], issues: List[str],
                         root=None, converged=False,
                         derivative_str: str = "") -> str:
        """Answer a student's question about the solver results."""
        if not question or not question.strip():
            return "Please type a question first."
        
        if not self.available:
            return "AI Assistant is not connected. Add GOOGLE_API_KEY or AI_GEN_API_KEYS in the .env file to enable answers."
        
        # Build iteration summary (first 3 + last 2)
        iter_summary = self._format_history(history)
        
        # Build context
        context = f"Function: f(x) = {function_str}\n"
        if derivative_str:
            context += f"Derivative: f'(x) = {derivative_str}\n"
        if root is not None:
            context += f"Root found: x = {root}\n"
        context += f"Converged: {'Yes' if converged else 'No'}\n"
        if issues:
            context += f"Issues: {', '.join(issues)}\n"
        context += f"Iterations summary:\n{iter_summary}\n"
        
        prompt = f"""You are helping a student understand a Newton-Raphson solver project.
Answer in simple student-friendly language.

{context}

Student question: {question}

Give a clear answer in 4-8 sentences. If the question is unrelated to this Newton-Raphson solver project, politely say that this assistant is focused on the Newton-Raphson solver."""
        
        try:
            return self._try_generate_content(prompt)
        except Exception:
            return "AI Assistant could not connect with the available API keys. Please check the keys or try again later."
    
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
            return {
                "guesses": [],
                "reasoning": "AI Advisor could not connect with the available API keys. Please check the keys or try again later."
            }