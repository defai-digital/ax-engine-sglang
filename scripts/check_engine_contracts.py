"""Verify versioned contract assets and, optionally, their AX Engine mirror."""
import argparse
import hashlib
import json
from pathlib import Path


def check(root, manifest):
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"contract asset differs: {root / name}: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer", type=Path, help="AX Engine checkout containing the fixture mirror")
    parser.add_argument("--result", type=Path, help="benchmark record to validate (requires jsonschema)")
    parser.add_argument("--validate-schema", action="store_true", help="check JSON Schema syntax (requires jsonschema)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "contracts"
    manifest = json.loads((root / "manifest.json").read_text())
    check(root, manifest)
    if args.validate_schema or args.result:
        from jsonschema import Draft202012Validator
        schema = json.loads((root / "benchmark-result-v1.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        if args.result:
            Draft202012Validator(schema).validate(json.loads(args.result.read_text()))
    if args.peer:
        peer = args.peer / "crates/ax-engine-core/tests/fixtures"
        check(peer, manifest)
        if json.loads((peer / "engine-contract-manifest.json").read_text()) != manifest:
            raise SystemExit("peer contract manifest differs")
    print("Engine contract assets match their pinned SHA-256 digests" + (" in both repositories" if args.peer else ""))


if __name__ == "__main__":
    main()
