import argparse
import ast

from app.scaffold import run_app


def _parse_scalar(value: str):
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v in {"{}", "[]"}:
        return ast.literal_eval(v)
    if v.startswith("[") and v.endswith("]"):
        return ast.literal_eval(v)
    try:
        if any(ch in v for ch in [".", "e", "E"]):
            return float(v)
        return int(v)
    except ValueError:
        return v


def simple_yaml_load(path: str):
    root = {}
    stack = [(0, root)]
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip() or raw_line.strip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.strip()

            while stack and indent < stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if rest == "":
                parent[key] = {}
                stack.append((indent + 2, parent[key]))
            else:
                parent[key] = _parse_scalar(rest)
    return root


def parse_args():
    parser = argparse.ArgumentParser(description="Run JEPA app")
    parser.add_argument("--fname", required=True, help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = simple_yaml_load(args.fname)
    run_app(cfg)


if __name__ == "__main__":
    main()
