# 📐 AI-Guided Newton-Raphson Solver

An interactive web application that solves **f(x) = 0** using the Newton-Raphson iterative method, with optional AI-powered guidance from Google Gemini.

Built with **Streamlit**, **SymPy**, **Matplotlib**, and the **Google GenAI SDK**.

---

## Features

- **Symbolic differentiation** — enter any function of `x` and the derivative is computed automatically.
- **Step-by-step iteration table** — inspect every Newton-Raphson step (x, f(x), f'(x), error).
- **Live visualisation** — function plot with tangent lines and a log-scale convergence chart.
- **AI Advisor** — get AI-powered starting suggestions, failure diagnosis, and step explanations directly in the advisor panel.
- **Global AI Assistant (Chatbot)** — a premium floating chatbot accessible from any tab in the top-right corner. It uses solver results to answer context-specific questions about your equations, or fallback prompts for general Newton-Raphson theory.
- **Automatic Key Rotation** — load one or multiple Gemini API keys. The app automatically rotates keys upon failure to prevent quota/rate limits.
- **Pre-loaded examples** — one-click common functions to try.

---

## Prerequisites

- **Python 3.10+**

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/erojbajracharya/AI-Guided-Newton-Raphson-Solver.git
cd AI-Guided-Newton-Raphson-Solver
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API keys (optional — for AI features)

Create a `.env` file in the project root.

You can configure a single key:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

Or configure multiple keys for automatic rotation:
```env
AI_GEN_API_KEYS=api_key_one,api_key_two,api_key_three
```

You can get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

> **Note:** The solver works perfectly without an API key — AI features will simply be disabled.

---

## Running the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at **http://localhost:8501**.

---

## Usage

1. Enter a function in the sidebar (e.g. `x**2 - 4`, `cos(x) - x`, `exp(x) - 3*x`).
2. Set an initial guess, tolerance, and max iterations.
3. Click **🚀 Solve**.
4. Explore the results across four tabs:
   - **📊 Visualization** — function plot with tangent lines and convergence chart.
   - **📋 Iterations** — detailed table of every step.
   - **🤖 AI Advisor** — get AI-powered suggestions and failure diagnosis.
   - **📖 Method Info** — learn how Newton-Raphson works.
5. Ask context-aware questions to the **💬 Ask AI** floating assistant in the top-right corner.

---

## How to Present This Project

To present this project to an audience or in a student presentation, explain the workflow using these simple steps:

1. **User enters a mathematical function**: The user inputs any function $f(x)$ (like `x**2 - 4`) and a starting point $x_0$ using the sidebar.
2. **SymPy finds the derivative**: The application uses SymPy to calculate the exact symbolic derivative $f'(x)$ automatically.
3. **Newton-Raphson formula is applied repeatedly**: The solver executes the iterative formula:
   $$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$
   Each iteration refines the estimate of the root.
4. **Streamlit shows graph, iteration table, and convergence**: Streamlit renders an interactive visualization showing the curve, tangent lines, step log-scale error decrease, and a summary details table.
5. **Gemini AI is optional**: If a Google Gemini API Key is provided, the AI advisor provides mathematical insights. If not, the app uses built-in math rules to explain the iteration steps and diagnose issues.
6. **Key Rotator & Floating Assistant**: A custom `APIRotator` handles multiple keys seamlessly. A persistent chatbot widget floats at the top-right corner, letting users interact with the AI assistant from any tab context.

---

## Project Highlights

- **Automatic Differentiation**: Finds symbolic derivatives automatically using SymPy.
- **Iteration Table**: Displays a clean, tabular log of values for every step.
- **Graph Visualization**: Renders both function tangents and convergence history plots.
- **Convergence Checking**: Uses a dual check ensuring both $|f(x)|$ and $|x_{new} - x|$ are below tolerance.
- **Optional AI Guidance**: Leverages LLM expertise for tutoring and student-friendly explanations.
- **Robust API Key Rotator**: Seamlessly manages multiple API keys to avoid exhaustion.

---

## Project Structure

```
AI-Guided-Newton-Raphson-Solver/
├── app.py              # Streamlit web interface with global chatbot widget
├── solver.py           # Newton-Raphson solver engine
├── ai_advisor.py       # Google Gemini AI advisor services
├── api_rotator.py      # Google Gemini API key rotator
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore
└── README.md           # Documentation
```

---

## Supported Function Syntax

| Expression       | Syntax          |
| ---------------- | --------------- |
| Power            | `x**2`, `x**3`  |
| Square root      | `sqrt(x)`       |
| Trigonometric    | `sin(x)`, `cos(x)`, `tan(x)` |
| Exponential      | `exp(x)`        |
| Logarithm        | `log(x)`        |
| Constants        | `pi`, `E`       |
| Combinations     | `x**2 - sin(x) + 1` |

---