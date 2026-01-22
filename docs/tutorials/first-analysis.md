# Tutorial: Your First DocFrame Analysis

**Scope statement:** Walk through a minimal text‑analysis pipeline using DocFrame.

## Step 1 — Build a dataset

**Question:** *How do I create a small document dataset?*

**Answer:**

```python
import docframe as df

corpus = df.DocDataFrame.from_texts(
    texts=[
        "The quick brown fox jumps over the lazy dog.",
        "DocFrame makes text processing feel like data processing.",
    ],
    metadata={"source": ["sample-1", "sample-2"]},
)
```

## Step 2 — Clean and tokenize

**Question:** *How do I normalize the text before analysis?*

**Answer:**

```python
cleaned = corpus.clean_documents(lowercase=True, remove_punct=True)

tokens = cleaned.select([
    cleaned.active_document_name,
    cleaned.document.text.tokenize().alias("tokens"),
])
```

## Step 3 — Summarize

**Question:** *How do I get a quick summary of the corpus?*

**Answer:**

```python
summary = cleaned.describe_text()
```

## Step 4 — Explore counts

**Question:** *How do I compute word counts per document?*

**Answer:**

```python
with_counts = cleaned.add_word_count()
```

## Step 5 — Save results

**Question:** *How do I store the outputs?*

**Answer:**

```python
with_counts.write_parquet("analysis.parquet")
```

## Recap

**Question:** *What should I try next?*

**Answer:** Explore `docframe/docs/user-guide/text-operations.md` and experiment with n‑grams or concordance queries.
