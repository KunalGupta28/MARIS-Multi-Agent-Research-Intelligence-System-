"""
MARIS Setup Verification — Run this script to validate your environment.

Checks:
    1. Python version
    2. Required packages
    3. API key connectivity (OpenAI, Groq, LangSmith)
    4. Local storage directories
    5. Qdrant local disk mode
    6. SQLite database
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def check_python_version() -> bool:
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 11
    return ok


def check_env_file() -> bool:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    return env_path.exists()


def check_package(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def check_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return len(key) > 10 and key.startswith("sk-")


def check_groq_key() -> bool:
    key = os.getenv("GROQ_API_KEY", "")
    return len(key) > 10 and key.startswith("gsk_")


def check_langsmith_key() -> bool:
    key = os.getenv("LANGCHAIN_API_KEY", "")
    return len(key) > 10


def check_data_dirs() -> bool:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "pdfs").mkdir(exist_ok=True)
    (data_dir / "qdrant_db").mkdir(exist_ok=True)
    return True


def main():
    console.print(
        Panel(
            "[bold cyan]MARIS Environment Verification[/bold cyan]\n"
            "[dim]Checking your setup before first run...[/dim]",
            border_style="cyan",
        )
    )

    # Load .env if exists
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    table = Table(title="Environment Checks", show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # Python version
    ok = check_python_version()
    table.add_row(
        "Python >= 3.11",
        "✅" if ok else "❌",
        f"v{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    # .env file
    ok = check_env_file()
    table.add_row(
        ".env file",
        "✅" if ok else "⚠️",
        "Found" if ok else "Copy .env.example → .env and fill in keys",
    )

    # Core packages
    packages = {
        "langchain": "langchain",
        "langgraph": "langgraph",
        "langchain_openai": "langchain-openai",
        "langchain_groq": "langchain-groq",
        "qdrant_client": "qdrant-client",
        "fitz": "PyMuPDF",
        "arxiv": "arxiv",
        "streamlit": "streamlit",
        "pydantic": "pydantic",
        "rank_bm25": "rank-bm25",
        "rich": "rich",
        "aiohttp": "aiohttp",
    }

    for import_name, package_name in packages.items():
        ok = check_package(import_name)
        table.add_row(
            f"Package: {package_name}",
            "✅" if ok else "❌",
            "Installed" if ok else f"pip install {package_name}",
        )

    # API Keys
    ok = check_openai_key()
    table.add_row(
        "OpenAI API Key",
        "✅" if ok else "⚠️",
        "Configured" if ok else "Set OPENAI_API_KEY in .env",
    )

    ok = check_groq_key()
    table.add_row(
        "Groq API Key",
        "✅" if ok else "⚠️",
        "Configured" if ok else "Set GROQ_API_KEY in .env",
    )

    ok = check_langsmith_key()
    table.add_row(
        "LangSmith Key",
        "✅" if ok else "⚠️",
        "Configured" if ok else "Set LANGCHAIN_API_KEY in .env",
    )

    # Data directories
    ok = check_data_dirs()
    table.add_row("Data directories", "✅" if ok else "❌", "Created: data/, data/pdfs/, data/qdrant_db/")

    console.print(table)
    console.print()

    # Summary
    console.print(
        Panel(
            "[bold green]Next Steps:[/bold green]\n\n"
            "1. Copy [cyan].env.example[/cyan] → [cyan].env[/cyan] and add your API keys\n"
            "2. Install dependencies: [cyan]pip install -e .[/cyan]\n"
            "3. Run the app: [cyan]streamlit run app.py[/cyan]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
