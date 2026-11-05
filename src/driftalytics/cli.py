"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .core import MonitorRegistry
from .damage import DamagePredictor
from .runner import SlidingWindowRunner
from .version import __version__


def run_config(config_path: str) -> int:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    if "reference" not in config:
        raise ValueError("config requires a reference path")

    predictor = None
    if config.get("damage_predictor"):
        predictor = DamagePredictor.load(config["damage_predictor"])

    runner = SlidingWindowRunner(
        reference=config["reference"],
        monitors=config.get("monitors", ["psi"]),
        window_size=int(config.get("window_size", 1000)),
        alert_threshold=config.get("alert_threshold"),
        damage_predictor=predictor,
        stride=config.get("stride"),
        sliding=bool(config.get("sliding", False)),
        include_partial=bool(config.get("include_partial", True)),
    )
    report = runner.run(config.get("data", config["reference"]))
    print(report.summary())

    output = config.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping")
    if output.get("html"):
        report.to_html(output["html"])
    if output.get("json"):
        report.to_json(output["json"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="driftalytics", description="Driftalytics CLI")
    parser.add_argument("--version", action="version", version=f"driftalytics {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a YAML config")
    run_parser.add_argument("config", help="Path to YAML config")

    subparsers.add_parser("monitors", help="List registered monitors")
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return run_config(args.config)
        if args.command == "monitors":
            print("\n".join(MonitorRegistry.list()))
            return 0
        parser.print_help()
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
