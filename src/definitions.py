from pathlib import Path
import os

def resolve_project_root() -> Path:
    """Resolve the project root directory based on the location of this file."""
    PROJECT_ROOT = Path(__file__).resolve().parent
    while not (PROJECT_ROOT / ".git").exists():
        if PROJECT_ROOT.parent == PROJECT_ROOT:
            raise RuntimeError("Could not find project root containing .git")
        PROJECT_ROOT = PROJECT_ROOT.parent
    return PROJECT_ROOT

PROJECT_ROOT = resolve_project_root()


def enforce_absolute_path(file_path: str) -> str:
    """Ensure the given file path is absolute, resolving relative to project root if necessary."""
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


if __name__ == "__main__":
    # Example usage
    relative_path = "animals/lucy_v0.xml"
    absolute_path = enforce_absolute_path(relative_path)
    print(f"Absolute path: {absolute_path}")