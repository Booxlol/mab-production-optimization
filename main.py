"""Entry point for monoclonal antibody space-time profit optimization."""
from pathlib import Path
from src.analysis import run


if __name__ == "__main__":
    run(Path(__file__).resolve().parent)
