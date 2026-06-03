# TODO

Future work after the Survival Geometry and XGBoost/SHAP pipeline is stable:

- Tune XGBoost hyperparameters.
- Compare `core`, `clean`, and `full` feature sets with cross-validation.
- Add PCA or UMAP for visual geometry inspection.
- Test whether centroid-distance features improve prediction.
- Build a Titanic ontology graph for passenger, class, deck, family, ticket, and language relationships.
- Manually enrich `external/cabin_deck_map.csv` from historical deck plans.
- Run false-positive and false-negative deep analysis.
