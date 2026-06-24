"""
AI-Guided Newton-Raphson Solver
================================
Streamlit web interface for numerical root-finding with AI assistance.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

from solver import NewtonRaphsonSolver
from ai_advisor import GeminiAdvisor

# Page config
st.set_page_config(
    page_title="Newton-Raphson Solver",
    page_icon="📐",
    layout="wide"
)

# Load API key
load_dotenv()

# Custom CSS
st.markdown("""
<style>
    .success { color: #2ecc71; font-weight: bold; }
    .failure { color: #e74c3c; font-weight: bold; }
    .ai-box { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; color: white;
    }
</style>
""", unsafe_allow_html=True)


def set_example_function(expr):
    st.session_state["func"] = expr
    for key in ["result", "ai_guesses", "ai_diagnosis", "ai_step", "ai_assistant_answer"]:
        if key in st.session_state:
            del st.session_state[key]


def main():
    if "func" not in st.session_state:
        st.session_state["func"] = "x**2 - 4"

    st.title("📐 AI-Guided Newton-Raphson Solver")
    st.caption("Find roots of f(x) = 0 using iterative numerical methods")
    
    # Initialize AI advisor
    ai = GeminiAdvisor()
    
    # === SIDEBAR: INPUTS ===
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        func_input = st.text_input(
            "f(x) = ",
            key="func",
            help="Enter function using Python syntax: x**2, sin(x), exp(x), etc."
        )
        
        x0 = st.number_input(
            "Initial guess (x₀)",
            value=1.0,
            help="Starting point for iteration"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            tol = st.number_input("Tolerance", value=1e-10, format="%.0e")
        with col2:
            max_iter = st.number_input("Max iterations", value=100, min_value=1, max_value=1000)
        
        st.divider()
        
        solve_clicked = st.button("🚀 Solve", type="primary", use_container_width=True)
        
        if st.button("Clear Results", use_container_width=True):
            for key in ["result", "ai_guesses", "ai_diagnosis", "ai_step", "ai_assistant_answer"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
            
        st.divider()
        
        # Example functions
        st.subheader("📋 Examples")
        examples = {
            "x² - 4": "x**2 - 4",
            "x³ - x - 2": "x**3 - x - 2",
            "cos(x) - x": "cos(x) - x",
            "e^x - 3x": "exp(x) - 3*x",
            "x³ - 6x² + 11x - 6": "x**3 - 6*x**2 + 11*x - 6",
            "sin(x) - x/3": "sin(x) - x/3"
        }
        for name, expr in examples.items():
            st.button(
                name,
                key=f"ex_{name}",
                use_container_width=True,
                on_click=set_example_function,
                args=(expr,)
            )
        
        st.divider()
        
        ai_status = "✅ Connected" if ai.available else "⚠️ Not connected (add GOOGLE_API_KEY to .env)"
        st.caption(f"AI Status: {ai_status}")
    
    # === MAIN CONTENT ===
    
    if solve_clicked:
        # Validate function
        is_valid, validation_msg = NewtonRaphsonSolver.validate_function(func_input)
        if not is_valid:
            st.error(validation_msg)
        else:
            # Run solver
            with st.spinner("Computing..."):
                solver = NewtonRaphsonSolver(func_input, x0, tol, max_iter)
                root, converged, history, issues = solver.solve()
                plot_data = solver.get_plot_data()
            
            # Store in session state
            st.session_state['result'] = {
                'root': root,
                'converged': converged,
                'history': history,
                'issues': issues,
                'plot_data': plot_data,
                'func': func_input
            }
            
    # Welcome screen check
    if "result" not in st.session_state:
        st.markdown("""
## Welcome to the AI-Guided Newton-Raphson Solver

This app solves equations of the form **f(x) = 0** using the Newton-Raphson method.

### How to use:
1. Enter a function in the sidebar.
2. Choose an initial guess.
3. Set tolerance and maximum iterations.
4. Click **Solve** to start.

You can also click one of the example functions in the sidebar.
""")
        st.info("Enter a function or choose an example, then click Solve to begin.")
        return
        
    # Show derivative formula f'(x) to the user
    try:
        from sympy import sympify, diff, symbols, latex
        x_sym = symbols('x')
        f_sym = sympify(st.session_state['result']['func'])
        df_sym = diff(f_sym, x_sym)
        st.latex(rf"f'(x) = {latex(df_sym)}")
    except Exception as e:
        st.warning(f"Could not compute derivative: {e}")
    
    # === DISPLAY RESULTS ===
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        # Status banner
        if result['converged']:
            st.success(f"✅ Converged! Root found: x = {result['root']:.10f}")
        else:
            st.error(f"❌ Did not converge. Issues: {', '.join(result['issues'])}")
        
        # Summary Cards
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.metric(
                label="Root", 
                value=f"{result['root']:.6f}" if result['root'] is not None and np.isfinite(result['root']) else "N/A"
            )
        with col_c2:
            st.metric(
                label="Total Iterations", 
                value=len(result['history'])
            )
        with col_c3:
            final_err = result['history'][-1]['error'] if result['history'] else 0.0
            st.metric(
                label="Final Error", 
                value=f"{final_err:.2e}" if result['history'] else "N/A"
            )
        with col_c4:
            status_val = "Converged" if result['converged'] else "Failed"
            st.metric(
                label="Status", 
                value=status_val
            )
            
        # === TABS ===
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visualization", 
            "📋 Iterations", 
            "🤖 AI Advisor",
            "📖 Method Info"
        ])
        
        # --- TAB 1: VISUALIZATION ---
        with tab1:
            col_plot1, col_plot2 = st.columns(2)
            
            with col_plot1:
                st.subheader("Function & Iterations")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                
                # Plot function
                x_plot = result['plot_data']['x']
                y_plot = result['plot_data']['y']
                ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label="f(x)")
                ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                
                # Plot tangents
                for i, tang in enumerate(result['plot_data']['tangents']):
                    alpha = 0.3 + 0.7 * (i / max(len(result['plot_data']['tangents']), 1))
                    ax1.plot(tang['x'], tang['y'], '--', color='orange', alpha=alpha, linewidth=1)
                
                # Plot iteration points
                iters = result['history']
                x_iter = [h['x'] for h in iters]
                y_iter = [h['f_x'] for h in iters]
                ax1.plot(x_iter, y_iter, 'ro-', markersize=8, label="Iterations")
                
                # Mark root
                if result['root'] is not None and np.isfinite(result['root']):
                    ax1.axvline(x=result['root'], color='green', linestyle=':', label=f"Root ≈ {result['root']:.4f}")
                
                ax1.set_xlabel("Input variable (x)", fontsize=10)
                ax1.set_ylabel("Function value f(x)", fontsize=10)
                ax1.set_title(f"Newton-Raphson Steps on f(x) = {result['func']}", fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                st.pyplot(fig1)
            
            with col_plot2:
                st.subheader("Error Convergence")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                
                iters_num = [h['iteration'] for h in result['history']]
                errors = [h['error'] for h in result['history']]
                
                ax2.semilogy(iters_num, errors, 'g-o', markersize=8, linewidth=2)
                ax2.axhline(y=tol, color='red', linestyle='--', label=f"Tolerance = {tol}")
                ax2.set_xlabel("Iteration Number", fontsize=10)
                ax2.set_ylabel("Error Magnitude |f(x)|", fontsize=10)
                ax2.set_title("Convergence Rate (Error per Iteration)", fontsize=12, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig2)
        
        # --- TAB 2: ITERATIONS TABLE ---
        with tab2:
            st.subheader("Iteration Details")
            st.write("Each row shows one Newton-Raphson iteration.")
            
            # Format data for table
            table_data = []
            for h in result['history']:
                table_data.append({
                    "Iter": h['iteration'],
                    "x": f"{h['x']:.10f}",
                    "f(x)": f"{h['f_x']:.2e}",
                    "f'(x)": f"{h['df_x']:.4f}",
                    "Step Size": f"{h['step_size']:.2e}",
                    "|Error|": f"{h['error']:.2e}"
                })
            
            st.dataframe(table_data, use_container_width=True, hide_index=True)
            
            # Summary stats
            st.divider()
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Iterations", len(result['history']))
            with col_b:
                st.metric("Final Error", f"{result['history'][-1]['error']:.2e}")
            with col_c:
                if len(result['history']) > 1:
                    st.metric("Last Step Size", f"{result['history'][-1]['step_size']:.2e}")
        
        # --- TAB 3: AI ADVISOR ---
        with tab3:
            st.subheader("🤖 AI-Powered Guidance")
            
            if not ai.available:
                st.warning("AI Advisor is not connected. Add an API key in .env to enable AI guidance.")
            else:
                # AI Suggestion buttons
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                
                with col_btn1:
                    if st.button("💡 Suggest Starting Points", use_container_width=True):
                        with st.spinner("Asking AI..."):
                            suggestions = ai.suggest_initial_guesses(result['func'])
                            st.session_state['ai_guesses'] = suggestions
                
                with col_btn2:
                    if not result['converged']:
                        if st.button("🔍 Diagnose Failure", use_container_width=True):
                            with st.spinner("Analyzing..."):
                                diagnosis = ai.diagnose_failure(
                                    result['func'], result['history'], result['issues']
                                )
                                st.session_state['ai_diagnosis'] = diagnosis
                    else:
                        st.button("🔍 Diagnose Failure", disabled=True, use_container_width=True)
                
                with col_btn3:
                    if st.button("📝 Explain Last Step", use_container_width=True):
                        with st.spinner("Generating explanation..."):
                            last = result['history'][-1]
                            prev = result['history'][-2] if len(result['history']) > 1 else None
                            explanation = ai.explain_step(result['func'], last, prev)
                            st.session_state['ai_step'] = explanation
                
                # Display AI responses
                st.divider()
                
                if 'ai_guesses' in st.session_state:
                    guesses = st.session_state['ai_guesses']
                    st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                    st.markdown("### 💡 Suggested Initial Guesses")
                    if 'guesses' in guesses and guesses['guesses']:
                        for i, g in enumerate(guesses['guesses'], 1):
                            st.markdown(f"**Guess {i}:** x₀ = {g}")
                    if 'reasoning' in guesses:
                        st.markdown(f"*{guesses['reasoning']}*")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if 'ai_diagnosis' in st.session_state:
                    st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                    st.markdown("### 🔍 Failure Diagnosis")
                    st.markdown(st.session_state['ai_diagnosis'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if 'ai_step' in st.session_state:
                    st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                    st.markdown("### 📝 Step Explanation")
                    st.markdown(st.session_state['ai_step'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Click to explain any iteration
                st.divider()
                st.subheader("Click any iteration to get AI explanation")
                selected = st.selectbox(
                    "Select iteration:",
                    options=range(len(result['history'])),
                    format_func=lambda i: f"Iteration {result['history'][i]['iteration']}"
                )
                if st.button("Explain This Iteration"):
                    step = result['history'][selected]
                    prev = result['history'][selected-1] if selected > 0 else None
                    with st.spinner("Thinking..."):
                        exp = ai.explain_step(result['func'], step, prev)
                    st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                    st.markdown(exp)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # === AI ASSISTANT Q&A ===
            st.divider()
            st.subheader("💬 Ask AI Assistant")
            
            if not ai.available:
                st.warning("AI Assistant is not connected. Add an API key in .env to enable question answering.")
            else:
                # Example question buttons
                eq1, eq2, eq3 = st.columns(3)
                with eq1:
                    if st.button("Why did it converge?", key="eq_converge", use_container_width=True):
                        st.session_state["ai_question"] = "Why did it converge?"
                with eq2:
                    if st.button("Explain the final iteration", key="eq_final", use_container_width=True):
                        st.session_state["ai_question"] = "Explain the final iteration"
                with eq3:
                    if st.button("What does tolerance mean?", key="eq_tol", use_container_width=True):
                        st.session_state["ai_question"] = "What does tolerance mean?"
                
                # Question input
                question = st.text_area(
                    "Ask a question about the method, result, graph, or iterations:",
                    value=st.session_state.get("ai_question", ""),
                    placeholder="Example: Why did the method converge so fast?",
                    key="ai_question_input"
                )
                
                if st.button("Ask AI Assistant", use_container_width=True, type="primary"):
                    # Get derivative string if available
                    deriv_str = ""
                    try:
                        from sympy import sympify, diff, symbols
                        x_sym = symbols('x')
                        f_sym = sympify(result['func'])
                        deriv_str = str(diff(f_sym, x_sym))
                    except Exception:
                        pass
                    
                    with st.spinner("Thinking..."):
                        answer = ai.answer_question(
                            function_str=result['func'],
                            question=question,
                            history=result['history'],
                            issues=result['issues'],
                            root=result['root'],
                            converged=result['converged'],
                            derivative_str=deriv_str
                        )
                        st.session_state['ai_assistant_answer'] = answer
                
                # Display answer
                if 'ai_assistant_answer' in st.session_state:
                    st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                    st.markdown("### 💬 AI Assistant Answer")
                    st.markdown(st.session_state['ai_assistant_answer'])
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # --- TAB 4: METHOD INFO ---
        with tab4:
            st.markdown("""
            ## Newton-Raphson Method
            
            ### Formula
            $$x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}$$             
            ### How it Works
            1. Start with initial guess x₀
            2. Calculate f(x₀) and f'(x₀)
            3. Draw tangent line at (x₀, f(x₀))
            4. Find where tangent crosses x-axis → that's x₁
            5. Repeat until |f(xₙ)| < tolerance
            
            ### Convergence
            - **Order:** Quadratic (error squares each step near root)
            - **Condition:** f'(x) ≠ 0 near root, good initial guess
            - **Rate:** |eₙ₊₁| ≈ M|eₙ|² where M = |f''(ξ)/(2f'(ξ))|
            
            ### Advantages
            - Very fast convergence (quadratic)
            - Simple to implement
            - Works well with good initial guess
            
            ### Limitations
            - Requires derivative f'(x)
            - Fails if f'(x) = 0
            - Sensitive to initial guess
            - Can oscillate or diverge
            
            ### Common Failure Modes
            | Issue | Cause | Solution |
            |-------|-------|----------|
            | Zero derivative | f'(xₙ) = 0 | Different starting point |
            | Oscillation | Alternating sides of root | Try bisection method |
            | Divergence | Starting too far from root | Better initial guess |
            | Slow convergence | Near inflection point | Increase iterations |
            """)


if __name__ == "__main__":
    main()