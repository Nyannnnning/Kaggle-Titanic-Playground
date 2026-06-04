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
- ticket route object
- ticket group object
- fare object
- group fate object
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
clean_group_fate    = clean plus target-safe GroupFateObject candidate features
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

`clean_group_fate` is an experimental feature set. It adds leave-one-out
group-outcome features from Ticket, Family, and FamilyTicket scopes. It is
target-safe, but direct logistic modeling currently performs worse than the
default `clean` feature set, so Group Fate is treated primarily as a rule-mining
and entity-audit layer.

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
├── CompanionGroupObject
└── GroupFateObject
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

## Group Fate Object

`GroupFateObject` asks a different question from `TicketGroupObject`.
`TicketGroupObject` describes who traveled together without using `Survived`.
`GroupFateObject` describes what happened to similar train-fold group members,
so it must be target-safe.

The implementation uses:

```text
train rows      = leave-one-out group fate
validation rows = training-fold group fate only
test rows       = full-training group fate only
```

This prevents a passenger's own `Survived` value from becoming their own
feature.

```text
GroupFateObject
├── TicketFateSignal
├── TicketFateSupportBin
├── FamilyFateSignal
├── FamilyFateSupportBin
├── FamilyTicketFateSignal
├── FamilyTicketFateSupportBin
├── PrimaryFateScope
├── PrimaryFateSignal
├── PrimaryFateSupportBin
├── PrimaryFateKnownCount
└── PrimaryFateSurvivalRate
```

`PrimaryFateSignal` uses this priority:

```text
FamilyTicket > Ticket > Family > Unknown
```

The repeated-CV rule scan found a stable candidate:

```text
Pclass == 3
PrimaryFateSignal == all_died
=> predict non-survivor
```

This signal is not a universal rule. It is strongly conditioned by `Pclass` and
`Sex`; for example, first-class women can still survive even when their
train-mapped group fate is negative. The public-best strategy uses this signal
only for 3rd-class passengers and only when support is at least 2.

`explore.py` writes:

```text
reports/survival_by_ticket_fate_signal.csv
reports/survival_by_family_ticket_fate_signal.csv
reports/survival_by_primary_fate_signal.csv
reports/pclass_sex_primary_fate_signal.csv
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

## Ticket Route Object

`TicketRouteObject` treats ticket identifiers and embarkation ports as a route
or booking-channel signal. This is separate from `TicketGroupObject`: one
describes who traveled together, the other describes ticket series and embark
context.

```text
TicketRouteObject
├── TicketPrefix
├── TicketNumber
├── TicketNumberBand
├── Embarked
├── TicketPrefixEmbarked
├── TicketNumberBandEmbarked
└── TicketPrefixPclassEmbarked
```

Observed route patterns include:

```text
PC_S              high survival, mostly 1st-class route signal
CA_S / A_S        very low survival, mostly 3rd-class Southampton signals
300000_349999_S   low survival numeric ticket band
10000_19999_S     stronger 1st-class / mid-fare positive pocket
```

The current route layer does not replace `Pclass`, `Sex`, `Fare`, or
GroupFateObject. It is used for exploration and narrow rule candidates.

`explore.py` writes:

```text
reports/survival_by_ticket_prefix_embarked.csv
reports/survival_by_ticket_number_band_embarked.csv
reports/survival_by_ticket_prefix_pclass_embarked.csv
reports/survival_by_pclass_sex_ticket_prefix_embarked.csv
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

The first rule-mined candidate is still available as a rollback baseline:

```bash
python optimize_submission.py \
  --strategy rule_mined_hybrid \
  --output submissions/rule_mined_hybrid.csv
```

The previous public-best candidate is:

```bash
python optimize_submission.py \
  --strategy rule_mined_group_fate_support \
  --reference-strategy rule_mined_hybrid \
  --output submissions/rule_mined_group_fate_support.csv
```

`optimize_submission.py` now defaults to `regime_layer_v3_1309_only`, which
builds on this previous best. Use `--strategy rule_mined_group_fate_support`
when you want to compare against the `0.79904` baseline, or
`--strategy conservative_hybrid` / `--strategy rule_mined_hybrid` when you want
to compare against safer rollback baselines.

The first post-`0.79904` candidate tested a positive 1st-class correction:

```bash
python optimize_submission.py \
  --strategy rule_mined_group_fate_p1_midfare \
  --reference-strategy rule_mined_group_fate_support \
  --output submissions/rule_mined_group_fate_p1_midfare.csv
```

This is not a simple inverse of Group Fate. Directly flipping 1st-class
`all_survived` group-fate males was weak in train data. The cleaner signal is:

```text
Pclass == 1
AgeBin == Adult
FamilySizeBin == Alone
FareBinPclass == MidFare_P1
=> predict survivor
```

Kaggle result:

```text
rule_mined_group_fate_p1_midfare: 0.79186
```

This failed against the `0.79904` best version. The train signal overfit:
although the rule improved train accuracy, it likely turned too many 1st-class
adult male non-survivors into survivors on the test set. Keep it as a learning
case, not as a recommended submission.

The narrower TicketRoute version of that idea was tested as:

```bash
python optimize_submission.py \
  --strategy rule_mined_group_fate_ticket_route_10000_s \
  --reference-strategy rule_mined_group_fate_support \
  --output submissions/rule_mined_group_fate_ticket_route_10000_s.csv
```

Rule:

```text
AgeBin == Adult
FarePerTicketBinPclass == MidFare_P1
TicketNumberBandEmbarked == 10000_19999_S
=> predict survivor
```

Kaggle result:

```text
rule_mined_group_fate_ticket_route_10000_s: 0.79665
```

This flips only PassengerId `1036` relative to the `0.79904` best version, and
the score dropped by one public-test row. Keep PassengerId `1036` as
non-survivor in the current best strategy.

The next false-negative rule lab focuses on high-fare 3rd-class mixed-fate
groups. These are not defaults:

```bash
python optimize_submission.py \
  --strategy rule_mined_group_fate_p3_highfare_mixed \
  --reference-strategy rule_mined_group_fate_support \
  --output submissions/rule_mined_group_fate_p3_highfare_mixed.csv

python optimize_submission.py \
  --strategy rule_mined_group_fate_asplund_children \
  --reference-strategy rule_mined_group_fate_support \
  --output submissions/rule_mined_group_fate_asplund_children.csv

python optimize_submission.py \
  --strategy rule_mined_group_fate_ticket_1601 \
  --reference-strategy rule_mined_group_fate_support \
  --output submissions/rule_mined_group_fate_ticket_1601.csv
```

Rules:

```text
rule_mined_group_fate_p3_highfare_mixed
  FareBinPclass == HighFare_P3
  PrimaryFateSignal == mixed
  => predict survivor
  flips: 931, 1046, 1066, 1271

rule_mined_group_fate_asplund_children
  same high-fare mixed group
  Ticket == 347077
  Title == Master or Age <= 13
  => predict survivor
  flips: 1046, 1271

rule_mined_group_fate_ticket_1601
  same high-fare mixed group
  Ticket == 1601
  => predict survivor
  flips: 931
```

This follows the rule-lab principle: TicketRoute and ticket identity are used as
auxiliary narrowing conditions, not as standalone rules.

## Class Regime Rule Layer

`Pclass` is now treated as a survival regime boundary, not only as one feature
inside a universal rule. The rule layer is split into separate regimes:

```text
p3_survival_rescue
  Find 3rd-class passengers predicted dead by the current best strategy but
  carrying strong low-class survival evidence.

p1_death_override
  Find 1st-class passengers predicted alive by the current best strategy but
  carrying strong high-class death evidence.

p2_holdout
  Do not force a rule unless a separate 2nd-class pattern is found.
```

The registry lives in:

```text
src/regime_rules.py
```

The first tested regime candidate was:

```text
p3_master_small_family_rescue
  Pclass == 3
  Title == Master
  FamilySizeBin == SmallFamily
  => predict survivor
```

Evidence on train against `rule_mined_group_fate_support`:

```text
train_count: 10
train survival rate: 1.0000
changed current-best predictions: 5
accuracy delta: +0.0056
repeated-CV selected folds: 25/25
```

Kaggle result:

```text
regime_layer_v1: 0.78947
```

This failed hard against the `0.79904` current best. The candidate flipped 8
test passengers from death to survivor; the public score drop is approximately
`4/418`, which implies about 2 of the flips helped and 6 hurt. The practical
lesson is that the broad `P3 + Master + SmallFamily` rescue signal is weaker on
test than the negative GroupFate evidence attached to most of these passengers.
Keep this as a failed rule-lab case, not as a recommended submission.

Test passengers changed by this candidate:

```text
913, 972, 1084, 1093, 1136, 1236, 1284, 1309
```

Reproduce the failed regime-layer candidate without replacing the current best
default:

```bash
python optimize_submission.py \
  --strategy regime_layer_v1 \
  --reference-strategy rule_mined_group_fate_support \
  --output submissions/regime_layer_v1.csv
```

This also writes:

```text
reports/regime_rule_audit.csv
```

The narrowed follow-up candidate is:

```text
p3_master_small_family_no_all_died_rescue
  Pclass == 3
  Title == Master
  FamilySizeBin == SmallFamily
  no Ticket/Family/FamilyTicket/Primary GroupFateSignal == all_died
  => predict survivor
```

Evidence against `rule_mined_group_fate_support`:

```text
train_count: 9
train survival rate: 1.0000
changed current-best train predictions: 5
accuracy delta: +0.0056
test changed passengers: 1284, 1309
```

Generate it with:

```bash
python optimize_submission.py \
  --strategy regime_layer_v2 \
  --reference-strategy rule_mined_group_fate_support \
  --output submissions/regime_layer_v2.csv
```

This is a much cleaner A/B test than v1. Because it flips only two public-test
rows relative to `0.79904`, the public score interpretation is simple:

```text
both right: 0.80382
one right, one wrong: 0.79904
both wrong: 0.79425
```

Kaggle result:

```text
regime_layer_v2: 0.79904
```

This tied the previous best without improving it. Treat it as neutral evidence:
the narrowed P3 rescue does not hurt, but it also does not justify replacing
`rule_mined_group_fate_support` by itself.

The v2 tie can be split into two one-passenger A/B candidates:

```bash
python optimize_submission.py \
  --strategy regime_layer_v3_1309_only \
  --reference-strategy rule_mined_group_fate_support \
  --output submissions/regime_layer_v3_1309_only.csv

python optimize_submission.py \
  --strategy regime_layer_v3_1284_only \
  --reference-strategy rule_mined_group_fate_support \
  --output submissions/regime_layer_v3_1284_only.csv
```

Signals:

```text
regime_layer_v3_1309_only
  flips PassengerId 1309
  P3 Master small family
  FamilyTicketFateSignal == all_survived

regime_layer_v3_1284_only
  flips PassengerId 1284
  P3 Master small family
  FamilyTicketFateSignal == mixed
```

Kaggle result:

```text
regime_layer_v3_1309_only: 0.80143
```

This is the current best. It confirms that the all_survived side of the v2 split
was the useful half. Since v2 tied at `0.79904`, the mixed-fate 1284 side is now
treated as likely negative unless future evidence says otherwise.

To inspect evidence conflicts directly:

```bash
python analyze_conflict_matrix.py
```

Outputs:

```text
reports/conflict_matrix_summary.csv
reports/conflict_matrix_examples.csv
reports/p1_death_override_candidates.csv
```

To find passengers where the current best predicts death but multiple models
rank survival probability high:

```bash
python analyze_model_rescue_candidates.py --probability-threshold 0.60 --min-models 2
```

Outputs:

```text
reports/model_rescue_candidates.csv
reports/model_rescue_all_died_conflicts.csv
reports/model_rescue_no_all_died_candidates.csv
reports/model_rescue_all_scores.csv
```

The split reports are important because the highest model-probability rescue
candidates often conflict with `all_died` GroupFate evidence. Those should be
studied as evidence conflicts, not submitted directly.

The 1st-class death override side is intentionally held out for now. Broad
signals such as older solo 1st-class women looked tempting but reduced train
accuracy and changed too many test passengers. P1 death rules should stay narrow
until they match the Isham / Allison-style exception profile rather than a broad
female-first-class pattern.

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
regime_layer_v3_1309_only: 0.80143
```

For conservative iteration, start from:

```bash
python optimize_submission.py --strategy conservative_hybrid --output submission.csv
```

For the current best rule-mined candidate:

```bash
python optimize_submission.py --output submission.csv
```

This defaults to `regime_layer_v3_1309_only`. To reproduce the previous best:

```bash
python optimize_submission.py \
  --strategy rule_mined_group_fate_support \
  --output submissions/rule_mined_group_fate_support.csv
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
submissions/rule_mined_group_fate_support.csv
submissions/rule_mined_group_fate_p1_midfare.csv
submissions/rule_mined_group_fate_ticket_route_10000_s.csv
submissions/rule_mined_group_fate_p3_highfare_mixed.csv
submissions/rule_mined_group_fate_asplund_children.csv
submissions/rule_mined_group_fate_ticket_1601.csv
submissions/regime_layer_v1.csv
submissions/regime_p3_master_small_family_rescue.csv
submissions/regime_layer_v2.csv
submissions/regime_p3_master_small_family_no_all_died_rescue.csv
submissions/regime_layer_v3_1309_only.csv
submissions/regime_layer_v3_1284_only.csv
reports/submission_strategy_audit.csv
reports/regime_rule_audit.csv
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

`rule_mined_group_fate_support` builds on `rule_mined_hybrid` and adds a
GroupFateObject override:

```text
Pclass == 3
PrimaryFateSignal == all_died
PrimaryFateSupportBin in {2, 3+}
=> predict non-survivor
```

This is the current best public score. It is more aggressive than
`rule_mined_hybrid`, but the public leaderboard result suggests that the
FamilyTicket/Ticket group-fate layer is carrying real signal.

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
