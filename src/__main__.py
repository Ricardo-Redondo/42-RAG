from fire import Fire
from pydantic import ValidationError
from src.cli import Cli
import sys


def main() -> int:
    Fire(Cli)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValidationError as e:
        print(f"Invalid argument: {e.errors()[0]['msg']}")
    except KeyboardInterrupt:
        print("\033[43mInterrupted aborting...\033[0m")