from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .model import get_classifier, get_feature_names


def logistic_coefficients_frame(model) -> pd.DataFrame:
    classifier = get_classifier(model)
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("logistic_coefficients_frame requires a LogisticRegression model.")

    coefficients = classifier.coef_[0]
    feature_names = get_feature_names(model)
    directions = np.select(
        [coefficients > 1e-9, coefficients < -1e-9],
        ["positive", "negative"],
        default="neutral",
    )

    coefficients_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "direction": directions,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    return coefficients_df.sort_values("abs_coefficient", ascending=False)


def write_logistic_coefficients(model, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients_df = logistic_coefficients_frame(model)
    coefficients_df.to_csv(output_path, index=False)
    return coefficients_df
