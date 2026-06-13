# 🤝 Contributing to MARIS

Thank you for your interest in contributing to the Multi-Agent Research Intelligence System (MARIS)! 

By contributing to this project, you help build a premium, autonomous tool for scientific literature research. This document provides guidelines for setting up your development environment, formatting your code, writing tests, and submitting PRs.

---

## 🚀 Development Setup

MARIS requires Python 3.11 or newer. We recommend setting up a virtual environment.

### 1. Clone & Initialize
```bash
# Clone the repository
git clone https://github.com/KunalGupta28/MARIS-Multi-Agent-Research-Intelligence-System-.git
cd "MARIS-Multi-Agent-Research-Intelligence-System-"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
Install the package in editable mode along with development dependencies:
```bash
pip install -e ".[dev]"
```

### 3. Environment Configuration
Copy the template and fill in your API keys (e.g. `GROQ_API_KEY`, `OPENAI_API_KEY`):
```bash
copy .env.example .env
```

---

## 🎨 Code Style & Standards

We enforce code quality standards using **Ruff**.

- **Line Length**: 120 characters maximum.
- **Imports**: Sorted and grouped automatically via Ruff's `isort` integration.
- **Type Annotations**: Mandatory for public methods, classes, and new function signatures.
- **Docstrings**: Required for all public modules, functions, classes, and methods. We follow the Google Style Python Docstrings standard.

### Running the Linter
Ensure your code is clean before committing:
```bash
# Lint check
python -m ruff check src/ tests/

# Automatically fix linting and formatting issues
python -m ruff check src/ tests/ --fix
```

---

## 🧪 Testing Guidelines

We use **pytest** to run unit and integration tests. All new features or bug fixes must include corresponding test coverage.

### Running Tests
Execute the entire test suite:
```bash
python -m pytest tests/ -v
```

### Custom Pytest Markers
We register three pytest markers to categorize tests:
- `unit`: Fast, lightweight unit tests with no external dependencies or database file disk side-effects (where mocks are fully utilized).
- `integration`: Tests that verify multi-component pipelines (e.g., SQLite interacting with PDF parser).
- `slow`: Heavy integration tests that download resources or execute long-running loops.

To run a specific subset of tests:
```bash
python -m pytest -m unit
```

---

## 📥 Pull Request Checklist

Before submitting a pull request, please make sure your branch adheres to the following:

1. [ ] **Format and Lint**: Run `python -m ruff check src/ tests/ --fix` and ensure no warnings remain.
2. [ ] **Tests Pass**: Run `python -m pytest` and make sure 100% of tests are passing.
3. [ ] **No Hardcoded Values**: All API keys, endpoints, and file paths must be resolved via `src/config.py` loaded from `.env`.
4. [ ] **Documentation**: Update `README.md` or write architectural notes under `docs/` if modifying core behaviors.
5. [ ] **Clean Git History**: Rebase against the main branch and write clear, concise commit messages.
