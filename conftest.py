"""pytest conftest — ensures the project root is on sys.path.

Without this, running `pytest` from any directory other than the project
root raises ModuleNotFoundError for `agents`, `models`, `utils`, etc.
"""
import sys
from pathlib import Path

# Insert the project root (the directory that contains this file) at the
# front of sys.path so that all first-party packages are importable.
sys.path.insert(0, str(Path(__file__).parent))
