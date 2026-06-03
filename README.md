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
- name object
- age object
- family object
- ticket object
- ticket group object
- fare object
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
│   ├── name_object.py
│   ├── age.py
│   ├── fare.py
│   ├── ticket_group.py
│   ├── spatial.py
│   ├── name_origin.py
│   ├── child_groups.py
│   ├── model.py
│   ├── explain.py
│   ├── geometry.py
│   ├── vectorizer.py
│   ├── entity.py
│   └── passenger_report.py
├── build_augmented_data.py
├── analyze_age_ablation.py
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

For XGBoost and SHAP experiments, install the optional dependency set:

```bash
pip install -r requirements-xgboost.txt
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
python train.py --model-type logistic_score --feature-set clean
python predict.py --input data/test.csv --output submission.csv
python explore.py
```

## Feature Sets

The default feature set is `clean`.

```text
core                = stable Titanic fields only
clean               = default; removes uncertain weak proxies
clean_group_objects = clean plus compact TravelGroupObject candidate features
full                = exploration mode; includes all provisional proxy signals
```

`clean` filters out:

```text
LifeboatAccessScore
StaircaseAccessScore
CabinZone
ClassArea
OriginInferenceConfidence
InferredOriginRegion
InferredPrimaryLanguage
EnglishComprehensionProxy
LanguageBarrierRisk
```

These features are not deleted from the entity layer. They are only excluded from the default model feature set because they are weak or provisional proxies.

`clean_group_objects` is an experimental feature set. It keeps the same clean
base and adds compact `CompanionGroupObject` fields. It is useful for controlled
ablation, but it is not the default because current CV evidence is mixed:
slightly higher repeated-CV accuracy, worse log loss, and no validation
accuracy gain on the fixed split.

## Advanced Features

The clean feature set includes stable engineered signals:

- `TicketGroupSize`
- `FamilyGroupSize`
- `AgeKnown`
- `AgeMissing`
- `AgeBin`
- `AgeMissingPatternByPclassTitle`
- `FareBin`
- `FamilySizeBin`
- `FarePerFamilyMember`
- `FarePerTicketMember`
- `FarePerFamilyTicketMember`
- `SexPclass`
- `TitlePclass`
- `AgeBinSex`
- `FareBinPclass`
- `FarePerTicketBinPclass`
- `TravelPartyType`
- `FareObjectInterpretation`
- `PclassAgeInteraction`
- `PclassFareInteraction`
- `IsChild`
- `IsAdultMale`
- `IsMother`
- `IsFirstClassFemale`
- `IsThirdClassMale`

`TicketGroupSize` and `FamilyGroupSize` are learned from the training fold and then applied to validation/test data, so validation does not compute its own local group map.

## Age Object

`Age` is treated as both an absolute life-stage signal and a class-relative
position. A 35-year-old passenger is not interpreted the same way in 1st class
and 3rd class, because each class has its own age distribution.

```text
AgeObject
├── Age
├── AgeKnown
├── AgeMissing
├── AgeBin
├── AgeZWithinPclass
├── AgeQuartileWithinPclass
├── AgeRelativeBucketWithinPclass
├── AgeRelativeBucketWithinPclassSex
├── AgeMissingPatternByPclassTitle
├── IsYoungWithinPclass
├── IsOldWithinPclass
├── IsOldMaleWithinPclass
└── IsYoungMasterWithinPclass
```

The class-relative age statistics are learned from the training fold and then
applied to validation/test rows. They are not recomputed on validation/test.

The default `clean` model keeps only the age-missing pattern from this object:

```text
AgeKnown
AgeMissing
AgeMissingPatternByPclassTitle
```

Ablation showed that the full class-relative age object was useful for analysis
but too noisy for the default logistic model. To rerun the comparison:

```bash
python analyze_age_ablation.py
```

Output:

```text
reports/age_ablation_scores.csv
reports/age_ablation_summary.csv
```

## Name Object

`NameObject` separates raw manifest names from formal-name analysis. Parentheses
are parsed for audit, but the default formal-name pattern reports ignore the
content inside parentheses.

```text
NameObject
├── FormalName
├── ParentheticalName
├── HasParentheticalName
├── FormalSurname
├── FormalTitle
├── FormalGivenNames
├── FormalGivenFirstToken
├── FormalGivenTokenCount
├── FormalGivenTokenCountBin
├── FormalNameHasInitial
├── FormalNameHasQuote
├── FormalSurnameShape
├── FormalNamePattern
└── FormalNameInterpretation
```

`Mrs. John Bradley (Florence Briggs Thayer)` is analyzed as formal name
`Mrs. John Bradley`; the parenthetical personal name is not used in formal-name
pattern summaries.

`explore.py` writes:

```text
reports/formal_name_pattern_summary.csv
reports/formal_name_title_token_summary.csv
reports/formal_name_surname_shape_summary.csv
reports/formal_name_parenthetical_presence_summary.csv
reports/formal_name_top_first_tokens.csv
reports/formal_name_examples.csv
```

## Travel Group Object

The entity layer now represents a passenger through several related objects:

```text
Passenger
├── FamilyGroupObject
├── FamilyTicketGroupObject
├── TicketGroupObject
├── FareObject
└── CompanionGroupObject
```

`FamilyKey = surname + Pclass` is treated as a weak grouping proxy. It is kept
for observation, but `CompanionGroupObject` uses a more conservative priority:

```text
family_ticket group > confirmed family group > ticket group > solo
```

The family group is only allowed to become the primary companion scope when the
passenger's original `FamilySize` is greater than 1. This avoids over-trusting
same-surname passengers who may not actually be traveling together.

Key fields:

```text
PrimaryCompanionScope
PrimaryGroupChildFemalePattern
ChildFemaleCompanionPattern
CompanionGroupSize
CompanionChildCount
CompanionFemaleCount
CompanionAdultMaleCount
CompanionAccompanyingChildCount
CompanionAccompanyingFemaleCount
CompanionAccompanyingAdultMaleCount
```

`explore.py` writes:

```text
reports/survival_by_primary_companion_pattern.csv
reports/survival_by_child_female_companion_pattern.csv
reports/survival_by_companion_count_bins.csv
reports/pclass3_child_female_companion_pattern.csv
reports/travel_group_object_examples.csv
```

## Ticket Group Object

`Ticket` is treated as a booking / travel-party identifier, not a passenger ID.
Many passengers share one ticket, and shared tickets can represent families,
non-family groups, or mixed travel parties.

```text
TicketGroupObject
├── TicketGroupSize
├── TicketChildCount
├── TicketFemaleCount
├── TicketMaleCount
├── TicketAdultMaleCount
├── TicketMasterCount
├── TicketDistinctFamilyCount
├── TicketMaxFamilySize
├── TicketChildRatio
├── TicketFemaleRatio
├── TicketAdultMaleRatio
├── TicketHasChild
├── TicketHasFemale
├── TicketHasAdultMale
├── TicketHasFamily
├── TicketIsMixedSex
├── TicketCompositionType
├── TicketFamilyPattern
└── TicketChildFemalePattern
```

These composition features are target-free, but the default `clean` logistic
model currently keeps them in the entity/report layer rather than the model
feature set. Direct inclusion made validation less stable. The actual survival
pattern of a ticket group uses `Survived`, so it is report-only and must not be
joined into test features.

`explore.py` writes:

```text
reports/ticket_group_survival_patterns.csv
reports/shared_ticket_group_summary_by_composition.csv
reports/shared_ticket_group_summary_by_family_pattern.csv
reports/shared_ticket_group_summary_by_child_female_pattern.csv
reports/survival_by_pclass_ticket_composition_type.csv
reports/survival_by_pclass_ticket_family_pattern.csv
reports/survival_by_pclass_ticket_child_female_pattern.csv
```

## Fare Object

`Fare` is treated as a ticket-level price field, not a clean passenger-level
wealth field. The project builds a `FareObject` from ticket and family context:

```text
FareObject
├── Fare
├── FareTotalPence
├── FarePounds
├── FareShillings
├── FarePence
├── FarePerTicketMember
├── FarePerFamilyMember
├── FarePerFamilyTicketMember
├── TravelPartyType
├── FareObjectInterpretation
├── TicketEqualsFamilySize
├── TicketFamilyMismatch
└── HighRawFareLargeFamilyFlag
```

The old British currency decomposition is kept for inspection only. It should
not be treated as a strong model signal by itself.

`Fare` is useful because it helps reconstruct ticket groups and travel-party
structure. It should not be interpreted as direct individual spending without
checking `TicketGroupSize`, `FamilySize`, and `FamilyTicketGroupSize`.

## Rule Mining

The project now supports rule discovery on top of an existing baseline strategy.
This is separate from model training. The goal is to find small, auditable
overrides such as:

```text
if Pclass == 3
and FareBin == LowFare
and TicketChildCount == 0
and Age is not <= 12
then predict non-survivor
```

Run:

```bash
python analyze_rule_candidates.py --baseline conservative_hybrid
```

Outputs:

```text
reports/rule_candidates_conservative_hybrid.csv
reports/rule_cv_scores_conservative_hybrid.csv
reports/rule_cv_summary_conservative_hybrid.csv
```

The current first rule-mined candidate is available as:

```bash
python optimize_submission.py \
  --strategy rule_mined_hybrid \
  --output submissions/rule_mined_hybrid.csv
```

This strategy is intentionally not the default. It is a candidate for manual
submission testing because it is more aggressive than `conservative_hybrid`.

## Cross-Validation

Training runs 5-fold stratified cross-validation by default:

```bash
python train.py --model-type logistic_score --feature-set clean --cv-folds 5
python train.py --model-type xgboost --feature-set clean --cv-folds 5
```

Output:

```text
reports/cross_validation_scores.csv
```

Use `--cv-folds 0` to skip cross-validation during quick experiments.

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

### `xgboost`

Optional nonlinear model with SHAP analysis:

```bash
python train.py --model-type xgboost --feature-set clean
```

Outputs:

```text
reports/shap_summary.csv
reports/passenger_shap_values.csv
reports/passenger_shap_force_report.csv
```

For `xgboost`, `reports/passenger_force_report.csv` uses SHAP positive and negative forces.

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

## Optimized Submission

The current best known public score is tracked in:

```text
kaggle_score_log.csv
```

Current best submitted strategy:

```text
rule_mined_hybrid: 0.78708
```

For conservative iteration, start from:

```bash
python optimize_submission.py --strategy conservative_hybrid --output submission.csv
```

For the current best rule-mined candidate:

```bash
python optimize_submission.py --strategy rule_mined_hybrid --output submission.csv
```

`conservative_hybrid` starts from the gender baseline and only applies high-confidence overrides:

- predict death for 3rd-class female passengers in large families
- predict survival for male `Master` passengers in 1st/2nd class
- predict survival for young 3rd-class male `Master` passengers in small families

Outputs:

```text
submission.csv
submissions/gender_baseline.csv
submissions/conservative_hybrid.csv
submissions/child_count_hybrid.csv
submissions/optimized_hybrid.csv
submissions/rule_mined_hybrid.csv
reports/submission_strategy_audit.csv
reports/submission_override_audit.csv
reports/submission_strategy_delta_audit.csv
```

`child_count_hybrid` is a narrow experiment around `Sex x Pclass=3 x child group`
signals. It keeps the conservative strategy and additionally tests 3rd-class male
`Master` passengers up to age 12 in small families.

`optimized_hybrid` has higher train accuracy but is more aggressive and should be treated as an experiment.

`rule_mined_hybrid` applies a repeated-CV rule-mined override:

```text
Pclass == 3
FareBin == LowFare
TicketChildCount == 0
Age is not <= 12
=> predict non-survivor
```

It is the current best public score, but it is more aggressive than
`conservative_hybrid` because it flips a group of 3rd-class low-fare female
passengers to non-survivor.

## Child Group Exploration

`src/child_groups.py` adds exploratory child-count context:

- `IsChildProxy`
- `TicketChildCountTotal`
- `TicketAccompanyingChildCount`
- `FamilyKeyChildCountTotal`
- `FamilyKeyAccompanyingChildCount`
- `AnyGroupChildCountTotal`
- `AnyAccompanyingChildCount`

`FamilyKey` is a weak proxy because unrelated passengers can share surname and
class. Treat ticket-based child counts as the more conservative signal.

`explore.py` writes:

```text
reports/pclass3_sex_any_accompanying_child_count.csv
reports/pclass3_sex_ticket_accompanying_child_count.csv
reports/pclass3_sex_family_size_any_accompanying_child_count.csv
reports/pclass3_title_age_family_child_window.csv
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
├── name
├── age
├── family
├── ticket
├── ticket_group
├── fare
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
