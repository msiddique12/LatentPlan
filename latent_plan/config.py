from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def apply_config_to_namespace(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> argparse.Namespace:
    known = {action.dest for action in parser._actions}  # noqa: SLF001
    unknown = sorted(k for k in config.keys() if k not in known)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    for key, value in config.items():
        setattr(args, key, value)
    return args


def parse_args_with_optional_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    cfg_path = getattr(args, "config", "")
    if not cfg_path:
        return args

    config = load_json_config(cfg_path)
    return apply_config_to_namespace(parser=parser, args=args, config=config)

