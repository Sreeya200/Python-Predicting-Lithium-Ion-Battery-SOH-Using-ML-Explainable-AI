import argparse
import subprocess
import sys
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STEPS = [
    ("00", "00_build_dataset.py"),
    ("01", "01_EDA.py"),
    ("02", "02_RandomForest.py"),
    ("03", "03_XGBoost.py"),
    ("04", "04_SHAP.py"),
    ("05", "05_model_evaluation.py"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Step numbers to skip, e.g. --skip 01 03")
    parser.add_argument("--only", nargs="*", default=[],
                        help="Run only these step numbers, e.g. --only 02 04")
    return parser.parse_args()


def main():
    args = parse_args()
    valid = {number for number, _ in STEPS}

    unknown = (set(args.skip) | set(args.only)) - valid
    if unknown:
        sys.exit(f"Unknown step number(s): {sorted(unknown)}. "
                 f"Valid steps: {sorted(valid)}")

    selected = [
        (number, name) for number, name in STEPS
        if (not args.only or number in args.only) and number not in args.skip
    ]

    total_start = time.time()

    for number, name in selected:
        print("\n" + "=" * 70)
        print(f"STEP {number}: {name}")
        print("=" * 70, flush=True)

        start = time.time()
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / name)],
                                cwd=PROJECT_ROOT)

        if result.returncode != 0:
            sys.exit(f"\nStep {number} ({name}) failed "
                     f"with exit code {result.returncode}.")

        print(f"\nStep {number} finished in {time.time() - start:.1f}s")

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {time.time() - total_start:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
