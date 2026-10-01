"""Aida CLI. Climate calculator for building renovations.

Usage (bin/aida runs "python -m aida.cli" with .env and the project's venv):
    bin/aida intake --input "<description>"
    bin/aida baseline --project <project.json>
    bin/aida alternatives --project <project.json> --baseline <baseline.json>
    bin/aida aggregate --project <project.json> --selections <selections.json>
    bin/aida report --project <project.json> --selections <selections.json> [--format markdown|pdf]
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    if command in ("-h", "--help", "help"):
        print(__doc__)
        return
    # Remove the command from argv so submodules see their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "intake":
        from aida.agents.intake import main as intake_main
        intake_main()
    elif command == "baseline":
        from aida.agents.baseline import main as baseline_main
        baseline_main()
    elif command == "alternatives":
        from aida.agents.alternatives import main as alternatives_main
        alternatives_main()
    elif command == "aggregate":
        from aida.agents.aggregate import main as aggregate_main
        aggregate_main()
    elif command == "report":
        from aida.agents.report import main as report_main
        report_main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
