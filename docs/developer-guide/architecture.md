# DocFrame Architecture (Developer Guide)

**Scope statement:** This page summarizes DocFrame’s internal architecture for contributors.

## 1) Core building blocks

**Question:** *What are the primary classes?*

**Answer:**

- `DocDataFrame` (eager) and `DocLazyFrame` (lazy).
- A document column mixin that preserves metadata.
- The `.text` namespace registered on Polars expressions/series/frames.

## 2) Text namespace pattern

**Question:** *How does `.text` work under the hood?*

**Answer:** DocFrame registers namespace classes with Polars (`@pl.api.register_*_namespace`) and forwards operations to shared utilities.

## 3) I/O wrapping

**Question:** *How does DocFrame preserve the document column during I/O?*

**Answer:** The `docio` decorator wraps Polars readers/writers and converts outputs into `DocDataFrame` or `DocLazyFrame` while tracking the active document column.

## 4) Extension guidance

**Question:** *Where do I add a new text operation?*

**Answer:**

1. Add the core implementation to `core/text_utils.py`.
2. Expose it in the `.text` namespace classes.
3. Add tests in `docframe/tests/` that validate eager and lazy behavior.

## Recap

**Question:** *What else should a contributor read?*

**Answer:** The API overview is in `docframe/docs/reference/api-overview.md` and the test guidance sits in the developer guide’s testing section (to be added as the suite evolves).
