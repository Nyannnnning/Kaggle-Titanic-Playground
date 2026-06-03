from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_INPUT_PATH = Path("data/test.csv")
DEFAULT_MODEL_PATH = Path("models/titanic_model.pkl")
DEFAULT_OUTPUT_PATH = Path("submission.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Titanic Kaggle submission.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to test.csv. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to saved model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the submission CSV. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_df = pd.read_csv(args.input)
    if "PassengerId" not in test_df.columns:
        raise ValueError(f"{args.input} must contain a PassengerId column.")

    model = joblib.load(args.model)
    predictions = model.predict(test_df).astype(int)

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": predictions,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    print(f"Submission saved to: {args.output}")


if __name__ == "__main__":
    main()
