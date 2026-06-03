# titanic-cli

A local CLI project for Kaggle's Titanic - Machine Learning from Disaster competition.

This project is no longer centered on Random Forest. The main direction is a first-pass **Survival Geometry Model**:

- predict Kaggle submissions
- build an interpretable survival score
- inspect positive and negative feature pressure
- place passengers in a high-dimensional geometry
- compare each passenger to survivor and non-survivor prototypes
- analyze clusters and boundary cases

The project treats each passenger as an entity, not just a CSV row. A passenger is assembled from multiple objects:

- identity object
- family object
- ticket object
- spatial object
- language / origin proxy object
- model evidence object
- geometry object

These objects are projected into a model-ready Passenger Vector.

## Project Structure

```text
titanic-cli/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── gender_submission.csv
│   └── derived/
│       ├── spatial_features.csv
│       └── name_origin_features.csv
├── external/
│   ├── deck_layout_features.csv
│   ├── cabin_deck_map.csv
│   └── README.md
├── models/
│   └── titanic_model.pkl
├── reports/
│   ├── validation_scores.csv
│   ├── logistic_coefficients.csv
│   ├── survival_geometry_validation.csv
│   ├── passenger_clusters.csv
│   ├── passenger_vectors.csv
│   ├── passenger_vector_features.csv
│   ├── passenger_feature_contributions.csv
│   ├── passenger_force_report.csv
│   ├── passenger_profiles/
│   ├── cluster_summary.csv
│   ├── prototype_profiles.json
│   └── exploratory_summary.md
├── src/
│   ├── features.py
│   ├── spatial.py
│   ├── name_origin.py
│   ├── model.py
│   ├── explain.py
│   ├── geometry.py
│   ├── vectorizer.py
│   ├── entity.py
│   └── passenger_report.py
├── build_augmented_data.py
├── train.py
├── predict.py
├── explore.py
├── requirements.txt
├── TODO.md
└── .gitignore
```

## Setup

```bash
cd titanic-cli
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Put Kaggle's Titanic CSV files in `data/`:

```text
data/train.csv
data/test.csv
data/gender_submission.csv
```

## First Full Pipeline

```bash
python build_augmented_data.py
python train.py --model-type logistic_score
python predict.py --input data/test.csv --output submission.csv
python explore.py
```

## Model Types

### `logistic_score`

Default model.

Uses:

- `LogisticRegression`
- `StandardScaler` for numeric features
- `OneHotEncoder` for categorical features
- `ColumnTransformer`
- `Pipeline`

This model is used to create an interpretable positive/negative survival score.

Output:

```text
reports/logistic_coefficients.csv
```

### `hist_gradient_boosting`

Optional stronger nonlinear model using sklearn's built-in:

```text
HistGradientBoostingClassifier
```

No XGBoost dependency is required.

### `random_forest_optional`

Optional comparison model only.

```bash
python train.py --model-type random_forest_optional
```

## Prediction

`predict.py` does not need to know the model type. It loads:

```text
models/titanic_model.pkl
```

and writes a Kaggle-compatible file:

```csv
PassengerId,Survived
892,0
893,1
894,0
```

## Survival Score

For `logistic_score`, `SurvivalScore` is the model decision score. Positive values push toward survivor space; negative values push toward non-survivor space.

Validation output:

```text
reports/validation_scores.csv
```

## Geometry

`src/geometry.py` extracts the model's preprocessed feature matrix and computes:

- survivor centroid
- non-survivor centroid
- distance to survivor centroid
- distance to non-survivor centroid
- survival distance delta
- KMeans cluster ID

Output:

```text
reports/survival_geometry_validation.csv
reports/survival_geometry_train.csv
reports/passenger_clusters.csv
reports/prototype_profiles.json
```

`SurvivalDistanceDelta` is:

```text
DistanceToNonSurvivorCentroid - DistanceToSurvivorCentroid
```

Larger values mean the passenger is closer to survivor space.

## Passenger Entity Reports

Training writes entity-level outputs:

```text
reports/passenger_vectors.csv
reports/passenger_vector_features.csv
reports/passenger_feature_contributions.csv
reports/passenger_force_report.csv
reports/passenger_profiles/passenger_<PassengerId>.json
```

`passenger_force_report.csv` is the compact view:

```text
PassengerId
SurvivalProbability
SurvivalScore
DistanceToSurvivorCentroid
DistanceToNonSurvivorCentroid
SurvivalDistanceDelta
ClusterId
IsBoundaryCase
PositiveForces
NegativeForces
```

Each JSON profile has the object-layer structure:

```text
Passenger Entity
├── identity
├── family
├── ticket
├── spatial
├── language_origin
├── model_evidence
├── geometry
└── forces
```

## Spatial Features

`src/spatial.py` creates provisional spatial proxy features:

- `CabinKnown`
- `CabinMissing`
- `Deck`
- `DeckKnown`
- `DeckMissing`
- `CabinNumber`
- `DeckOrdinal`
- `ApproxVerticalDistanceToBoatDeck`
- `SpatialAccessTier`
- `ClassDeckConsistency`

Optional external templates:

```text
external/deck_layout_features.csv
external/cabin_deck_map.csv
```

These are proxies only. They are not exact walking distance, exact lifeboat access, or final historical cabin reconstruction.

## Name Origin Features

`src/name_origin.py` creates conservative proxy features:

- `InferredOriginRegion`
- `InferredPrimaryLanguage`
- `EnglishComprehensionProxy`
- `LanguageBarrierRisk`
- `OriginInferenceConfidence`

These are not personal identity claims. They are weak proxy features for exploratory modeling.

## Future Work

Do not add these in the first pass:

- XGBoost
- SHAP
- passenger-level contribution decomposition
- PCA / UMAP
- Titanic ontology graph
- full manual cabin layout reconstruction
