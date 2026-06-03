# External Titanic Spatial Data

This folder holds optional spatial enrichment files for the Titanic CLI project.

The current files are intentionally conservative templates. They are proxy features for analysis, not verified cabin-level reconstructions.

## Sources To Review

- Encyclopedia Titanica deck plans: https://www.encyclopedia-titanica.org/titanic-deckplans/
- Encyclopedia Titanica Boat Deck notes: https://www.encyclopedia-titanica.org/titanic-deckplans/boat-deck.html
- Encyclopedia Titanica cabin allocations: https://www.encyclopedia-titanica.org/cabins.html

## Provisional Assumptions

- `DeckOrdinal` follows this rough vertical order: Boat, A, B, C, D, E, F, G, Unknown.
- `ApproxVerticalDistanceToBoatDeck` currently equals `DeckOrdinal`.
- This is not exact walking distance.
- This is not exact lifeboat access.
- `LifeboatAccessScore` and `StaircaseAccessScore` are rough deck-level proxies.
- Cabin assignment records are incomplete and partly reconstructed from historical fragments.
- `ClassDeckConsistency` is a modeling proxy, not a historical truth claim.

## Files

`deck_layout_features.csv` contains deck-level proxy values.

`cabin_deck_map.csv` is a blank template for future manual cabin-level review. Do not fill it without checking historical deck plans and documenting the source in `Notes`.
