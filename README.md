# 📐 AI-Guided Newton-Raphson Solver

An interactive web application that solves **f(x) = 0** using the Newton-Raphson iterative method, with optional AI-powered guidance from Google Gemini.

Built with **Streamlit**, **SymPy**, **Matplotlib**, and the **Google GenAI SDK**.

---

## Features

- **Symbolic differentiation** — enter any function of `x` and the derivative is computed automatically.
- **Step-by-step iteration table** — inspect every Newton-Raphson step (x, f(x), f'(x), error).
- **Live visualisation** — function plot with tangent lines and a log-scale convergence chart.
- **AI advisor** (optional) — uses Google Gemini to suggest starting points, diagnose failures, and explain individual iterations.
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

### 4. Configure the API key (optional — for AI features)

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

You can get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

> **Note:** The solver works perfectly without an API key — AI features will simply show fallback responses.

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

---

## Project Structure

```
AI-Guided-Newton-Raphson-Solver/
├── app.py              # Streamlit web interface
├── solver.py           # Newton-Raphson solver engine
├── ai_advisor.py       # Google Gemini AI integration
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed)
├── .gitignore
└── README.md
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

## License

This project is open source and available under the [MIT License](LICENSE).