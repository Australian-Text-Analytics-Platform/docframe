# Text Operations in DocFrame

**Scope statement:** This guide explains the most common `.text` operations and when to use them.

## 1) Tokenization and cleaning

**Question:** *How do I tokenize and clean text?*

**Answer:** Use the text namespace on expressions:

```python
import polars as pl
import docframe

cleaned = frame.select([
    pl.col(frame.active_document_name).text.clean().alias("cleaned"),
    pl.col(frame.active_document_name).text.tokenize().alias("tokens"),
])
```

## 2) Counts and summaries

**Question:** *How do I compute word or sentence counts?*

**Answer:**

```python
stats = frame.add_word_count().add_sentence_count()
```

## 3) N‑grams

**Question:** *How do I create bigrams or trigrams?*

**Answer:**

```python
bigrams = frame.select([
    pl.col(frame.active_document_name).text.ngrams(n=2).alias("bigrams"),
])
```

## 4) Pattern matching

**Question:** *How do I filter documents by a regex pattern?*

**Answer:**

```python
filtered = frame.filter_by_pattern(r"\b(data|model)\b")
```

## 5) Document‑term matrices

**Question:** *How do I create a document‑term matrix?*

**Answer:**

```python
dtm = frame.to_dtm(method="count")
```

## Recap

**Question:** *What should I read next?*

**Answer:** The reference page (`docframe/docs/reference/api-overview.md`) lists all supported operations, while the tutorial shows a full pipeline.
