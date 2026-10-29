from fire import Fire
from pydantic import ValidationError
from src.cli import Cli
from src.errors import RAGError
import sys


def main() -> int:
    """Run the CLI; turn every error into a message, never a traceback."""
    try:
        Fire(Cli)
    except RAGError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValidationError as e:
        err = e.errors()[0]
        field = ".".join(str(part) for part in err["loc"]) or "argument"
        print(f"Invalid argument '{field}': {err['msg']}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\033[43mInterrupted aborting...\033[0m", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
