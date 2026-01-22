# DocFrame API Overview (Reference)

**Scope statement:** A concise, non‑exhaustive overview of DocFrame’s public API surface.

## Core classes

**Question:** *Which classes should I know first?*

**Answer:**

- `DocDataFrame` — eager, document‑aware DataFrame.
- `DocLazyFrame` — lazy, document‑aware LazyFrame.

## I/O helpers

**Question:** *Which I/O helpers preserve the document column?*

**Answer:**

- `read_csv`, `read_parquet`, `read_json`, `read_excel`
- `scan_csv`, `scan_parquet`, `scan_ndjson`
- `read_text`, `read_zip`

## Text namespace (selected methods)

**Question:** *What are the most common `.text` operations?*

**Answer:**

- `tokenize()`
- `clean()`
- `word_count()`, `char_count()`, `sentence_count()`
- `ngrams(n=...)`
- `contains_pattern(pattern)`
- `concordance(...)`, `quotation()`

## Document utilities

**Question:** *How do I manage the document column?*

**Answer:** Use:

- `set_document(column)`
- `rename_document(new_name)`
- `filter_by_length(...)`
- `filter_by_pattern(...)`

## Recap

**Question:** *Where do I see this in context?*

**Answer:** The quickstart and tutorial pages show the same APIs in a full workflow.
