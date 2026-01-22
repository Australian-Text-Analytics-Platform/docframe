# DocFrame Quickstart

**Scope statement:** This guide shows the smallest end‑to‑end DocFrame workflow.

## Step 1 — Create a document‑aware frame

**Question:** *How do I create a DocDataFrame from raw texts?*

**Answer:**

```python
import docframe as df

frame = df.DocDataFrame.from_texts(
    texts=["Hello world", "DocFrame is fast"],
    metadata={"source": ["a", "b"]},
)
```

## Step 2 — Inspect the document column

**Question:** *How do I know which column is the document column?*

**Answer:** DocFrame tracks it for you:

```python
frame.active_document_name
```

## Step 3 — Run a text operation

**Question:** *How do I get word counts?*

**Answer:**

```python
with_counts = frame.add_word_count()
```

## Step 4 — Use the text namespace

**Question:** *What does `.text` do?*

**Answer:** The `.text` namespace exposes reusable text operations on expressions:

```python
import polars as pl

result = frame.select([
    pl.col(frame.active_document_name).text.tokenize().alias("tokens"),
    pl.col(frame.active_document_name).text.word_count().alias("word_count"),
])
```

## Step 5 — Save or scan data

**Question:** *How do I read or write files?*

**Answer:** Use DocFrame’s wrapped I/O helpers so the document column stays preserved:

```python
frame.write_parquet("documents.parquet")

lazy_frame = df.scan_csv("large.csv")
```

## Recap

**Question:** *What should I explore next?*

**Answer:** Read `docframe/docs/user-guide/text-operations.md` for more operations or jump to `docframe/docs/tutorials/first-analysis.md` for a guided walkthrough.
