"""
Newton-Raphson Method Solver
============================
Implements the iterative root-finding algorithm:
    x_{n+1} = x_n - f(x_n) / f'(x_n)

Features:
- Automatic symbolic differentiation
- Convergence detection
- Issue diagnosis
"""

import numpy as np
from sympy import sympify, diff, symbols
from typing import List, Dict, Tuple, Optional


class NewtonRaphsonSolver:
    """Newton-Raphson root-finding solver."""
    
    def __init__(self, function_str: str, x0: float, 
                 tol: float = 1e-10, max_iter: int = 100):
        """
        Initialize solver.
        
        Args:
            function_str: Mathematical expression (e.g., "x**2 - 4")
            x0: Initial guess
            tol: Convergence tolerance
            max_iter: Maximum iterations allowed
        """
        self.function_str = function_str
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        
        # Setup symbolic math
        self.x = symbols('x')
        self.f_sym = sympify(function_str)
        self.df_sym = diff(self.f_sym, self.x)
        
        # Create safe numeric evaluation helper
        def safe_eval(expr, val):
            try:
                res = expr.subs(self.x, val).evalf()
                # Check for complex numbers or symbol leftovers
                if res.is_real is False or res.is_real is None:
                    return float('nan')
                return float(res)
            except Exception:
                return float('nan')

        # Create numeric functions
        self.f = lambda val: safe_eval(self.f_sym, val)
        self.df = lambda val: safe_eval(self.df_sym, val)
        
        # Results
        self.history: List[Dict] = []
        self.converged: bool = False
        self.root: Optional[float] = None
        self.issues: List[str] = []

    def _add_issue(self, issue: str):
        """Add issue label if not already present to avoid duplicates."""
        if issue not in self.issues:
            self.issues.append(issue)

    def get_derivative_string(self) -> str:
        """Return string representation of the symbolic derivative."""
        return str(self.df_sym)
    
    def solve(self) -> Tuple[Optional[float], bool, List[Dict], List[str]]:
        """
        Execute Newton-Raphson iterations.
        
        Returns:
            Tuple of (root, converged, history, issues)
        """
        x = self.x0
        
        for i in range(self.max_iter):
            fx = self.f(x)
            dfx = self.df(x)
            
            # Check numerical stability of evaluation before using fx/dfx
            if not (np.isfinite(fx) and np.isfinite(dfx)):
                self._add_issue("NUMERICAL_INSTABILITY")
                break
            
            # Record iteration
            step_size = abs(fx / dfx) if dfx != 0 else float('inf')
            self.history.append({
                'iteration': i + 1,
                'x': x,
                'f_x': fx,
                'df_x': dfx,
                'step_size': step_size,
                'error': abs(fx)
            })
            
            # Check zero or small derivative
            if dfx == 0 or abs(dfx) < 1e-12:
                self._add_issue("ZERO_OR_SMALL_DERIVATIVE")
                break
            
            # Newton step
            x_new = x - fx / dfx
            
            # Check numerical stability of new step
            if not np.isfinite(x_new):
                self._add_issue("NUMERICAL_INSTABILITY")
                break
            
            # Check convergence using both:
            # a) abs(fx) < tolerance
            # b) abs(x_new - x) < tolerance
            if abs(fx) < self.tol and abs(x_new - x) < self.tol:
                self.converged = True
                self.root = x_new
                break
            
            x = x_new
        
        if not self.converged:
            self.root = x
            self._diagnose_issues()
        
        return self.root, self.converged, self.history, self.issues
    
    def _diagnose_issues(self):
        """Identify why solver failed to converge."""
        if len(self.history) >= self.max_iter:
            self._add_issue("MAX_ITERATIONS_REACHED")
        
        if len(self.history) > 2:
            # Detect oscillation (sign flipping)
            recent = self.history[-5:]
            sign_changes = sum(
                1 for i in range(1, len(recent))
                if np.sign(recent[i]['f_x']) != np.sign(recent[i-1]['f_x'])
            )
            if sign_changes >= 3:
                self._add_issue("OSCILLATING")
            
            # Detect divergence (errors increasing)
            errors = [h['error'] for h in recent]
            if all(errors[i] > errors[i-1] for i in range(1, len(errors))):
                self._add_issue("DIVERGING")
        
        # Check derivative at start
        if abs(self.df(self.x0)) < 1e-6:
            self._add_issue("SMALL_DERIVATIVE_AT_START")
    
    def get_plot_data(self) -> Dict:
        """Generate data for visualization."""
        if not self.history:
            return {'x': [], 'y': [], 'iterations': []}
        
        # Determine x range from iterations
        x_vals = [h['x'] for h in self.history]
        x_min = min(x_vals) - abs(min(x_vals)) * 0.5 - 1
        x_max = max(x_vals) + abs(max(x_vals)) * 0.5 + 1
        
        # Generate smooth curve
        x_plot = np.linspace(x_min, x_max, 300)
        y_plot = []
        
        for xp in x_plot:
            try:
                y = self.f(xp)
                if np.isfinite(y):
                    y_plot.append(y)
                else:
                    y_plot.append(None)
            except:
                y_plot.append(None)
        
        # Generate tangent lines for each iteration
        tangents = []
        for h in self.history:
            x_t = np.linspace(h['x'] - 2, h['x'] + 2, 50)
            y_t = [h['f_x'] + h['df_x'] * (xt - h['x']) for xt in x_t]
            tangents.append({'x': x_t.tolist(), 'y': y_t})
        
        return {
            'x': x_plot.tolist(),
            'y': y_plot,
            'tangents': tangents,
            'iterations': self.history,
            'root': self.root
        }
    
    @staticmethod
    def validate_function(func_str: str) -> Tuple[bool, str]:
        """Check if function string is valid."""
        try:
            x = symbols('x')
            sympify(func_str)
            return True, "Valid function"
        except Exception as e:
            return False, f"Invalid function: {str(e)}"