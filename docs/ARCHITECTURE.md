# DocFrame Architecture

## Overview

DocFrame is a GeoPandas-inspired text analysis library built on Polars, providing document-aware DataFrames with a unified text processing namespace. This document explains the architectural design, component interactions, and reusable patterns for AI tools.

## Design Philosophy

### Core Principles
1. **Lazy-by-default**: Chain operations without eager evaluation; materialize only when needed
2. **Composition over inheritance**: Wrap Polars objects while preserving their API
3. **Namespace registration**: Extend Polars with `.text` namespace for unified text operations
4. **Auto-detection**: Automatically identify document columns using heuristics
5. **Type preservation**: Operations return DocDataFrame/DocLazyFrame to maintain text-aware context

### GeoPandas Parallels
- **GeoPandas**: pandas + geometry column + spatial operations
- **DocFrame**: Polars + document column + text operations

## Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  DocDataFrame, DocLazyFrame, docio-wrapped I/O functions       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ├─────────────────────────────┐
                               │                             │
┌──────────────────────────────▼─────┐   ┌──────────────────▼─────────┐
│      Core Classes                  │   │   Text Namespace           │
│  - _DocumentColumnMixin            │   │  - TextExprNamespace       │
│  - DocDataFrame (eager)            │   │  - TextSeriesNamespace     │
│  - DocLazyFrame (lazy)             │   │  - TextDataFrameNamespace  │
│                                    │   │  - TextLazyFrameNamespace  │
│  Functions:                        │   │                            │
│  - guess_document_column()         │   │  Registered with Polars:   │
│  - from_texts()                    │   │  @pl.api.register_*        │
│  - serialize/deserialize           │   │                            │
└──────────────────────────────────┬─┘   └─────────────┬──────────────┘
                                   │                   │
                                   └─────────┬─────────┘
                                             │
                               ┌─────────────▼──────────────┐
                               │   Text Utilities           │
                               │  - tokenize()              │
                               │  - clean_text()            │
                               │  - word_count()            │
                               │  - char_count()            │
                               │  - sentence_count()        │
                               │  - concordance_elements()  │
                               │  - quotation_elements()    │
                               │  - compute_token_freq()    │
                               │  - extract_ngrams()        │
                               │  + NLTK/spaCy integration  │
                               └────────────────────────────┘
                                             │
                        ┌────────────────────┴───────────────────┐
                        │                                        │
          ┌─────────────▼──────────────┐      ┌────────────────▼────────────┐
          │  Polars Backend            │      │  Optional Dependencies      │
          │  - DataFrame/LazyFrame     │      │  - NLTK (tokenization)      │
          │  - Expression API          │      │  - spaCy (NLP)             │
          │  - Series operations       │      │  - scikit-learn (DTM/TFIDF) │
          │  - I/O functions           │      │  - BERTopic (topic model)   │
          └────────────────────────────┘      └─────────────────────────────┘
```

## Component Breakdown

### 1. Core Classes (`core/docframe.py`)

#### `_DocumentColumnMixin`
**Purpose**: Shared behavior for document column management  
**Used By**: DocDataFrame, DocLazyFrame  
**Key Methods**:
- `guess_document_column(df)`: Heuristic-based auto-detection (longest avg text length)
- `set_document(column)`: Switch active document column
- `rename_document(new_name)`: Rename document column
- `_validate_document_column(column)`: Ensure column exists and is string type

**Example**:
```python
# Auto-detect best text column in any DataFrame
column = DocDataFrame.guess_document_column(df)

# Validate before text operations
DocDataFrame._validate_document_column(df, column)
```

#### `DocDataFrame`
**Purpose**: Eager evaluation document-aware DataFrame  
**Inherits**: _DocumentColumnMixin  
**Wraps**: pl.DataFrame  
**Key Methods**:
- `from_texts(texts, metadata)`: Construct from text list + metadata
- `add_word_count()`, `add_char_count()`, `add_sentence_count()`: Add computed columns
- `clean_documents()`: Apply text cleaning to document column
- `filter_by_length()`, `filter_by_pattern()`: Text-based filtering
- `to_dtm()`: Create document-term matrix
- `describe_text()`: Comprehensive text statistics
- `serialize()`, `deserialize()`: Save/load with document metadata

**Example**:
```python
# Create from raw texts with metadata
doc_df = DocDataFrame.from_texts(
    texts=['doc1', 'doc2'],
    metadata={'source': ['a', 'b']}
)

# Quick text stats
stats = doc_df.describe_text()

# DTM for ML models
dtm = doc_df.to_dtm(method='tfidf')
```

#### `DocLazyFrame`
**Purpose**: Lazy evaluation document-aware DataFrame  
**Inherits**: _DocumentColumnMixin  
**Wraps**: pl.LazyFrame  
**Key Methods**:
- `collect()`: Materialize to DocDataFrame
- `fetch()`: Preview first N rows as DocDataFrame
- Mirrors DocDataFrame API but returns lazy results

**Example**:
```python
# Lazy text processing pipeline
result = (
    DocLazyFrame(large_df, document_column='content')
    .filter(pl.col('lang') == 'en')
    .with_columns(pl.col('content').text.clean())
    .collect()  # Execute only when needed
)
```

### 2. Text Namespace (`core/text_namespace.py`)

#### Four Namespace Classes (Polars Extension Pattern)

| Namespace Class          | Registers On       | Access Pattern         | Returns            |
|--------------------------|--------------------|------------------------|--------------------|
| `TextExprNamespace`      | pl.Expr            | `pl.col('text').text.*` | pl.Expr (lazy)     |
| `TextSeriesNamespace`    | pl.Series          | `series.text.*`        | pl.Series (eager)  |
| `TextDataFrameNamespace` | pl.DataFrame       | `df.text.*`            | pl.DataFrame       |
| `TextLazyFrameNamespace` | pl.LazyFrame       | `lf.text.*`            | pl.LazyFrame       |

**Registration Mechanism**:
```python
@pl.api.register_expr_namespace("text")
class TextExprNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr
```

**Shared Methods Across All Namespaces**:

- `tokenize(lowercase, remove_punct)`: Split text into tokens
  - **Implementation**: `TextExprNamespace` uses `map_elements()` with `partial(tokenize, ...)` from `text_utils`
  - **Delegation**: `TextSeriesNamespace` converts series to frame, calls expr method, converts back
  - **Pattern**: All namespace methods delegate to `text_utils` functions via expr operations
  
- `clean(lowercase, remove_punct, remove_digits, remove_extra_whitespace)`: Text normalization
  - **Implementation**: Uses `map_elements()` with `partial(clean_text, ...)` from `text_utils`
  - **Operations**: Applies lowercasing, punctuation removal (str.translate), digit removal (regex), whitespace normalization (regex)
  
- `word_count()`, `char_count()`, `sentence_count()`: Basic statistics
  - **Implementation**: Each uses `map_elements()` with corresponding `text_utils` function
  - **Methods**: `word_count()` uses `str.split()`, `char_count()` uses `len()`, `sentence_count()` uses regex `r"[.!?]+"`
  
- `ngrams(n)`: N-gram extraction
  - **Implementation**: Calls `extract_ngrams(text, n)` from `text_utils` via `map_elements()`
  - **Process**: Tokenizes text first, then creates sliding windows of n tokens
  
- `contains_pattern(pattern, case_sensitive)`: Regex matching
  - **Implementation**: Uses `map_elements()` with `contains_pattern()` from `text_utils`
  - **Method**: Applies `re.search()` with optional `re.IGNORECASE` flag
  
- `concordance(search_word, num_left_tokens, num_right_tokens)`: KWIC concordance
  - **Implementation**: Uses `map_elements()` with `concordance_elements()` from `text_utils`
  - **Returns**: List[Struct] with fields: `matched_word`, `l10`-`l1`, `r1`-`r10` (context tokens)
  - **Process**: Tokenizes, searches with regex, extracts left/right context tokens
  
- `quotation()`: Extract quoted text
  - **Implementation**: Uses `map_elements()` with `quotation_elements()` from `text_utils`
  - **Returns**: List[Struct] with fields: `quote_text`, `quote_type`, `token_count`, `is_floating_quote`
  - **Heuristic**: Detects paired quote marks (", ', «, etc.), handles nested quotes
  
- `remove_stopwords(stopwords)`: Filter stopwords
  - **Implementation**: Uses `map_elements()` with `remove_stopwords()` from `text_utils`
  - **Method**: Filters token list against stopwords set (case-insensitive)
  
- `join_tokens(separator)`, `filter_tokens(min_length)`: Token manipulation
  - **Implementation**: `join_tokens()` uses built-in `list.join()`, `filter_tokens()` uses `list.eval()` with length filter

**Namespace-Specific Methods**:

- **Series/Expr**: `to_dtm()` - Create DTM from tokenized column
  - **Implementation**: Calls `to_dtm_matrix()` from `text_utils` on tokenized series
  - **Process**: Collects vocabulary, creates term-document matrix using sklearn's CountVectorizer/TfidfVectorizer
  
- **DataFrame/LazyFrame**: Methods accept `column` parameter and create new columns with `{column}_{operation}` naming
  - **Pattern**: Use `with_columns(pl.col(column).text.{method}().alias(f"{column}_{suffix}"))`
  - **Delegation**: All operations delegate to expression-level namespace methods

**Example**:
```python
import polars as pl
import docframe  # Side-effect: registers text namespace

# Expression-level (use in select/with_columns)
df = df.select([
    pl.col('text').text.tokenize().alias('tokens'),
    pl.col('text').text.word_count().alias('word_count')
])

# Series-level (direct operations)
tokens = df['text'].text.tokenize()

# DataFrame-level (convenience)
df = df.text.add_word_count('text').text.clean('text')
```

#### Sequential Analysis (DataFrame + LazyFrame)

- **Purpose**: Bucket chronological or numeric values into evenly spaced periods, optionally grouped by categorical columns, and emit period boundaries plus counts.
- **Implemented In**: `TextDataFrameNamespace.sequential_analysis()` and `TextLazyFrameNamespace.sequential_analysis()`.
- **Key Inputs**:
  1. `time_column`: datetime/date or numeric column to bucket.
  1. `frequency`: `'hourly' | 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly'` (datetime mode only).
  1. `column_type`: `'datetime'` (default) or `'numeric'`.
  1. `numeric_origin` + `numeric_interval`: optional anchor/required bin width for numeric mode.
  1. `group_by_columns`: optional categorical splits appended after the time/numeric bin.
- **Execution Flow**:
  1. Normalize `column_type` and validate frequencies (datetime) or positive `numeric_interval` (numeric).
  1. Build `time_period`: Datetime columns use `.dt.truncate()` at the requested granularity (hourly, weekly, monthly, quarterly, yearly) while numeric columns cast to `Float64`, compute `(value - origin) / interval`, floor to `Int64`, and then project back to the left edge of the bin.
  1. `group_by(['time_period', *group_by_columns])` and aggregate `sequential_count`, `period_start`, `period_end`.
  1. Create `time_period_formatted`: Datetime mode uses `dt.strftime(time_format)` except for quarterly, which relies on `pl.format("{}-Q{}", year_expr, quarter_expr)`; numeric mode renders `[start, end)` using `pl.format("[{:.6g}, {:.6g})", start_expr, end_expr)` so integers drop trailing decimals while fractional bins stay precise.
  1. Sort chronologically (and by grouping columns) when `sort_by_time=True`.

##### Example – Hourly Buckets per Topic

```python
hourly = (
  df.text.sequential_analysis(
    time_column="created_at",
    group_by_columns=["topic"],
    frequency="hourly",
  )
)
```

##### Example – Numeric Sentiment Bins

```python
sentiment_bins = (
  df.text.sequential_analysis(
    time_column="sentiment_score",
    column_type="numeric",
    numeric_origin=-1.0,
    numeric_interval=0.25,
  )
)
```

##### Q & A

**Q:** What if `numeric_interval` is omitted when `column_type="numeric"`?

**A:** The namespace raises `ValueError("numeric_interval must be a positive number...")` because the bin width is required to render `[start, end)` labels.

### 3. Text Utilities (`core/text_utils.py`)

#### Core Processing Functions

- `tokenize(text, lowercase, remove_punct)`: NLTK word_tokenize wrapper
  - **Implementation**: Uses `nltk.tokenize.word_tokenize()` after optional lowercasing
  - **Dependencies**: NLTK punkt tokenizer (auto-downloaded via `_ensure_nltk_punkt()`)
  - **Filtering**: Removes punctuation-only tokens using `any(ch.isalnum() for ch in tok)`
  
- `clean_text(text, ...)`: Punctuation/digit removal, lowercasing, whitespace normalization
  - **Implementation**: Sequential transformations using str methods and regex
  - **Operations**: (1) `str.lower()`, (2) `str.translate()` for punctuation, (3) `re.sub(r"\d+")` for digits, (4) `re.sub(r"\s+")` for whitespace
  - **No dependencies**: Pure Python/regex implementation
  
- `word_count(text)`, `char_count(text)`, `sentence_count(text)`: Basic metrics
  - **Implementation**: `word_count()` uses `len(text.split())`, `char_count()` uses `len(text)`, `sentence_count()` uses `re.split(r"[.!?]+")`
  - **No dependencies**: Pure Python implementation
  
- `extract_ngrams(text, n)`: N-gram extraction
  - **Implementation**: Calls `tokenize()` first, then creates sliding windows: `[tokens[i:i+n] for i in range(len(tokens)-n+1)]`
  - **Dependencies**: Uses `tokenize()` from same module
  - **Returns**: List of space-joined n-gram strings
  
- `contains_pattern(text, pattern, case_sensitive)`: Regex search
  - **Implementation**: Uses `re.search(pattern, text, flags)` with optional `re.IGNORECASE`
  - **No dependencies**: Standard library regex only
- `remove_stopwords(tokens, stopwords)`: Stopword filtering

#### Advanced Analysis Functions

- `concordance_elements(text, search_word, num_left_tokens, num_right_tokens, regex, case_sensitive)`:
  - **Implementation**: Tokenizes text using `tokenize()`, searches for matches, extracts context windows
  - **Search**: Uses `re.search()` or `re.finditer()` with optional case-insensitive matching
  - **Context extraction**: Slices token list for left context `[max(0, idx-num_left):idx]` and right context `[idx+1:idx+1+num_right]`
  - **Returns**: List of dicts with matched word + left/right context tokens (l1-l10, r1-r10)
  - **Dependencies**: Uses `tokenize()` from same module
  
- `quotation_elements(text)`:
  - **Implementation**: Heuristic quotation detection using paired quote characters
  - **Algorithm**: Scans for opening quotes (", ', «, etc.), finds matching closing quotes, tracks nesting depth
  - **Validation**: Checks quote balance, identifies floating quotes (quotes without matching pairs)
  - **Returns**: List of dicts with quote_text, quote_type, token_count, is_floating_quote
  - **Dependencies**: Uses `tokenize()` for token counting
  
- `compute_token_frequencies(df, token_column, group_columns, method, normalize, min_freq, top_n)`:
  - **Implementation**: Explodes token column, groups by tokens (and optional group_columns), aggregates with `pl.len()`
  - **Normalization**: Optional division by total tokens per group
  - **Filtering**: Applies min_freq threshold and top_n sorting
  - **Returns**: Polars DataFrame with token counts/frequencies
  - **Dependencies**: Pure Polars operations (no external functions)

#### DTM/Topic Modeling Functions

- `to_dtm_matrix(tokens_series, method='count', min_df=1, max_df=1.0)`:
  - **Implementation**: Uses sklearn's `CountVectorizer` or `TfidfVectorizer` depending on method
  - **Process**: (1) Joins tokens back to strings, (2) Fits vectorizer, (3) Transforms to sparse matrix, (4) Converts to Polars DataFrame
  - **Parameters**: `min_df`/`max_df` filter vocabulary by document frequency
  - **Returns**: Tuple of (dtm_df: pl.DataFrame, vocabulary: list of terms)
  - **Dependencies**: sklearn.feature_extraction.text (CountVectorizer, TfidfVectorizer)
  
- `create_topic_model(dtm, n_topics, **kwargs)`:
  - **Implementation**: Uses BERTopic for topic modeling on document-term matrix
  - **Dimensionality reduction**: Optional UMAP if available, falls back to PCA
  - **Process**: (1) Reduces dimensions, (2) Clusters documents, (3) Extracts representative terms per topic
  - **Returns**: Tuple of (model, topics, probabilities)
  - **Dependencies**: BERTopic (optional), UMAP (optional, falls back to sklearn PCA)

#### NLTK Resource Management

- `_ensure_nltk_punkt()`: Auto-download punkt tokenizer
  - **Implementation**: Checks `nltk.data.find("tokenizers/punkt")`, downloads if missing
  - **Fallback**: Also attempts to download 'punkt_tab' for newer NLTK versions
  - **Caching**: Uses global `_NLTK_PUNKT_READY` flag to avoid repeated checks
  
- `_ensure_nltk_pos_tagger()`: Auto-download POS tagger
  - **Implementation**: Tries multiple tagger paths ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng')
  - **Fallback**: Continues trying alternative taggers if one fails
  - **Caching**: Uses global `_NLTK_POS_READY` flag
  
- `_get_sentence_tokenizer()`: Lazy-load sentence tokenizer
  - **Implementation**: Loads 'tokenizers/punkt/english.pickle' from NLTK data
  - **Fallback**: Creates default PunktSentenceTokenizer if pickle not found
  - **Caching**: Stores tokenizer in global `_PUNKT_SENTENCE_TOKENIZER` variable

**Example**:
```python
from docframe.core.text_utils import (
    tokenize, clean_text, compute_token_frequencies,
    concordance_elements, to_dtm_matrix
)

# Standalone text processing
tokens = tokenize("Hello world!", lowercase=True)
clean = clean_text("Text!!", remove_punct=True)

# Batch token frequency analysis
freq_df = compute_token_frequencies(
    df, 
    token_column='tokens', 
    group_columns=['category'],
    method='count',
    top_n=100
)

# KWIC concordance
concordances = concordance_elements(
    text, 
    search_word='example', 
    num_left_tokens=5, 
    num_right_tokens=5
)
```

### 4. I/O Utilities (`utils.py`)

#### `docio` Decorator

**Purpose**: Wrap Polars I/O functions to auto-detect document columns and return DocDataFrame/DocLazyFrame  

**Behavior**:
- `document_column=None` (default): Auto-detect using `guess_document_column()`
- `document_column='column_name'`: Use specified column
- `document_column=False`: Disable conversion, return plain Polars objects
- Falls back gracefully with warnings on errors

**Implementation**:
- Uses `functools.wraps` to preserve original function signature
- Pops `document_column` parameter from kwargs before calling underlying Polars function
- After getting result, checks type (DataFrame/LazyFrame) and converts if appropriate
- Calls `.text.to_docdataframe()` or `.text.to_doclazyframe()` via namespace methods
- Catches `ValueError` and `AssertionError`, issues warning, returns plain Polars object on failure

**Wrapped Functions**:
- `read_csv`, `scan_csv`
- `read_parquet`, `scan_parquet`
- `read_json`, `read_ndjson`, `scan_ndjson`
- `from_pandas`, `from_arrow`, `from_numpy`
- Conditionally: `read_excel`, `read_database`, `read_ipc`, `read_avro`, `read_delta`

**Example**:
```python
from docframe import read_csv, scan_parquet

# Auto-detect document column
doc_df = read_csv('data.csv')

# Explicit document column
doc_df = read_csv('data.csv', document_column='content')

# Disable for plain Polars DataFrame
df = read_csv('data.csv', document_column=False)

# Lazy reading with auto-detection
lazy_df = scan_parquet('large_data.parquet')
```

#### Custom Readers (`read_zip`, `read_text`)

While the majority of inputs come directly from Polars readers, DocFrame also ships
specialized helpers for raw text ingestion. Both functions reuse `docio`, so they
honor the same `document_column` contract and gracefully downgrade to plain Polars
objects when requested.

- `read_zip(path, *, encoding="utf-8", errors="ignore", text_extensions=None, include_extensionless=True)`
  - Streams textual members out of an archive, emitting one row per file with
    columns `file_path`, `base_name`, `extension`, and `document`.
  - Defaults the active document column to `document`, enabling downstream text
    namespace operations immediately while still exposing lightweight metadata
    for file provenance.

- `read_text(path, *, encoding="utf-8", errors="ignore")`
  - Materializes a single-document DocDataFrame from any plain-text file.
  - Always emits a minimalist one-column schema (`document`) so single-file
    ingestion mirrors the payload shape produced by archive reads.
  - Passing `document_column=False` returns a plain Polars DataFrame with the
    same schema for callers that prefer to opt out of DocDataFrame semantics.

**Example**:

```python
import docframe

doc_df = docframe.read_text("/data/notes.md")
assert doc_df.active_document_name == "document"

# Drop DocDataFrame semantics when only the raw document content is needed
raw_df = docframe.read_text(
  "/data/notes.md", document_column=False
)
assert raw_df.columns == ["document"]
```

## Data Flow Patterns

### Pattern 1: Document Column Auto-Detection

```text
User calls I/O function
    ↓
docio decorator intercepts
    ↓
Call Polars I/O → get DataFrame/LazyFrame
    ↓
Check document_column parameter:
  - None → guess_document_column(df)
  - 'column' → use column
  - False → skip conversion
    ↓
Convert to DocDataFrame/DocLazyFrame
    ↓
Return with active document column set
```

### Pattern 2: Text Namespace Method Call

```text
User calls df.select(pl.col('text').text.tokenize())
    ↓
Polars resolves pl.col('text').text
    ↓
TextExprNamespace.__init__(expr) stores self._expr
    ↓
tokenize() method called on namespace instance
    ↓
Creates partial function: partial(tokenize, lowercase=True, remove_punct=True)
    ↓
Wraps in expr.map_elements(_tokenize, return_dtype=pl.List(pl.String))
    ↓
Returns new Expr (lazy, not executed yet)
    ↓
Polars executes during collect() or other materialization
    ↓
map_elements applies tokenize() from text_utils to each cell
```

### Pattern 3: DocDataFrame Text Operation

```text
User calls doc_df.add_word_count()
    ↓
DocDataFrame checks self._document_column_name (active document column)
    ↓
Accesses document column: self.document (returns pl.Series)
    ↓
Calls series.text.word_count() via TextSeriesNamespace
    ↓
TextSeriesNamespace.word_count():
  - Converts series to frame: self._series.to_frame()
  - Calls pl.col(name).text.word_count() (delegates to TextExprNamespace)
  - TextExprNamespace uses map_elements with text_utils.word_count()
  - Converts back to series: .to_series()
    ↓
Result is pl.Series with word counts
    ↓
Adds to DataFrame: self._df.with_columns(word_counts.alias('word_count'))
    ↓
Returns new DocDataFrame with updated _df and same _document_column_name
```

### Pattern 4: DTM Creation Pipeline

```text
User calls doc_df.to_dtm(method='tfidf')
    ↓
DocDataFrame.to_dtm():
  1. Get document column Series: self.document
  2. Check if already tokenized (List[String] dtype)
  3. If not tokenized: apply .text.tokenize() via namespace
  4. Now have tokenized Series: pl.Series[List[String]]
    ↓
Call to_dtm_matrix(tokens_series, method='tfidf') from text_utils
    ↓
to_dtm_matrix() implementation:
  1. Join tokens back to strings: tokens.list.join(" ")
  2. Create sklearn vectorizer: TfidfVectorizer(min_df=min_df, max_df=max_df)
  3. Fit and transform: vectorizer.fit_transform(joined_texts)
  4. Get sparse matrix and vocabulary
  5. Convert sparse matrix to dense numpy array
  6. Create Polars DataFrame with vocabulary as columns
    ↓
Return (dtm_df: pl.DataFrame, vocabulary: List[str])
    ↓
DocDataFrame.to_dtm() returns dtm_df as Polars DataFrame
```

## Function Dependency Map

### High-Level Function Chains

#### Text Analysis Chain

```text
DocDataFrame.describe_text()
  → Calls self.document.text.word_count() (and other metrics)
    → TextSeriesNamespace.word_count()
      → Converts to frame, calls expr namespace
        → TextExprNamespace.word_count()
          → Uses map_elements(partial(word_count), return_dtype=pl.Int32)
            → text_utils.word_count(text)
              → Returns len(text.split())
```

**Key insight**: Series namespace always delegates to Expr namespace, which wraps text_utils functions in map_elements.

#### Token Frequency Analysis

```text
compute_token_frequencies(df, token_column='tokens', group_columns=['category'])
  → Validates inputs (DocDataFrame/DocLazyFrame)
  → Collects tokens from each frame
  → For each frame:
    - Get document series: frame.document or frame.collect().document
    - Tokenize if needed: doc_series.text.tokenize()
    - Flatten all tokens into single list
  → Creates union of all tokens across frames (universal vocabulary)
  → For each frame:
    - Count token frequencies using Counter
    - Ensure all tokens in vocabulary present (fill 0 for missing)
  → If 2 frames: compute log-likelihood statistics
    - Calculate expected frequencies
    - Compute G² statistic: 2 * Σ(observed * log(observed/expected))
    - Calculate Bayes Factor (BIC) and Effect Size (ELL)
  → Return (freq_dicts, stats_df)
```

**Key insight**: Uses pure Polars operations for aggregation, no external functions except for statistics.

#### DTM Creation

```text
DocDataFrame.to_dtm(method='tfidf', min_df=2, max_df=0.95)
  → Get document series: self.document
  → Tokenize if needed: 
    - Check dtype: if not List[String], call .text.tokenize()
    - Uses TextSeriesNamespace → TextExprNamespace → text_utils.tokenize
  → Call to_dtm_matrix(tokens_series, method='tfidf', min_df=2, max_df=0.95)
    → Join tokens to strings: tokens.list.join(" ")
    → Create sklearn TfidfVectorizer(min_df=2, max_df=0.95)
    → Fit and transform to sparse matrix
    → Convert to Polars DataFrame with vocabulary as columns
  → Return dtm_df
```

**Key insight**: Relies on sklearn for vectorization, converts between Polars and numpy/scipy formats.

#### Concordance Extraction

```text
pl.col('text').text.concordance('climate', num_left_tokens=10, num_right_tokens=10)
  → TextExprNamespace.concordance(search_word='climate', ...)
    → Creates wrapper function: _conc(text) that calls concordance_elements()
    → Wraps in map_elements(_conc, return_dtype=pl.List(pl.Struct([...])))
    → Returns lazy Expr
  → When executed (during collect):
    → For each text cell, calls concordance_elements(text, 'climate', 10, 10, ...)
      → Tokenizes text using text_utils.tokenize()
      → Searches for 'climate' in token list (with regex/case-sensitive options)
      → For each match, extracts context:
        - Left context: tokens[max(0, idx-10):idx]
        - Matched word: tokens[idx]
        - Right context: tokens[idx+1:idx+11]
      → Returns list of dicts with matched_word, l1-l10, r1-r10 fields
    → Result: Column of List[Struct] type
  → User can .explode() to get one row per concordance
  → User can .unnest() to unpack struct into columns
```

**Key insight**: Uses map_elements to apply concordance_elements() from text_utils, which delegates to tokenize().

### Initialization Dependencies

```text
import docframe
  → Executes src/docframe/__init__.py
    → Imports from core.docframe: DocDataFrame, DocLazyFrame
    → Imports from core.text_namespace (side-effect: namespace registration)
      → @pl.api.register_expr_namespace("text") registers TextExprNamespace
      → @pl.api.register_series_namespace("text") registers TextSeriesNamespace
      → @pl.api.register_dataframe_namespace("text") registers TextDataFrameNamespace
      → @pl.api.register_lazyframe_namespace("text") registers TextLazyFrameNamespace
    → Imports from core.text_utils: compute_token_frequencies, other utility functions
    → Imports from utils: docio-wrapped I/O functions (read_csv, scan_parquet, etc.)
    → Makes available in docframe namespace:
      - Classes: DocDataFrame, DocLazyFrame
      - Functions: compute_token_frequencies, read_csv, scan_parquet, from_pandas, etc.
```

**Key insight**: Simply importing docframe registers all namespace extensions with Polars via side-effects.

## Extension Points for AI Tools

### 1. Adding New Text Operations

**Where**: `core/text_namespace.py`  
**Pattern**:

```python
# In TextExprNamespace
def new_operation(self, param: str) -> pl.Expr:
    """New text operation"""
    from .text_utils import new_operation_impl
    
    _new_op = partial(new_operation_impl, param=param)
    return self._expr.map_elements(_new_op, return_dtype=pl.String)

# Mirror in TextSeriesNamespace
def new_operation(self, param: str) -> pl.Series:
    """New text operation"""
    return (
        self._series.to_frame()
        .select(pl.col(self._series.name).text.new_operation(param))
        .to_series()
    )

# Add DataFrame convenience wrapper
def new_operation(self, column: str, param: str) -> pl.DataFrame:
    """New text operation on column"""
    return self._df.with_columns(
        pl.col(column).text.new_operation(param).alias(f"{column}_new")
    )
```

### 2. Adding Custom Document Methods

**Where**: `core/docframe.py` in `DocDataFrame`  
**Pattern**:

```python
def custom_analysis(self) -> pl.DataFrame:
    """Custom document analysis"""
    doc_col = self.active_document_name
    return self._df.select([
        pl.col(doc_col).text.tokenize().alias('tokens'),
        # Custom aggregations
    ])
```

### 3. Adding New I/O Formats

**Where**: `utils.py`  
**Pattern**:

```python
from polars import read_custom_format as _read_custom
read_custom_format = docio(_read_custom)
```

### 4. Integrating ML Libraries

**Where**: `core/text_utils.py` or new module  
**Pattern**:

```python
def custom_model(df: DocDataFrame, **kwargs):
    """Custom ML pipeline"""
    # Get tokenized text
    tokens = df.document.text.tokenize()
    
    # Create DTM
    dtm, vocab = to_dtm_matrix(tokens, method='tfidf')
    
    # Apply custom model
    model = CustomModel(**kwargs)
    results = model.fit_transform(dtm)
    
    return results
```

## Performance Considerations

### Lazy vs Eager Execution

- **DocLazyFrame**: Use for large datasets, chains operations without execution
- **DocDataFrame**: Use when immediate results needed or dataset fits in memory
- **Best Practice**: Build pipelines with LazyFrame, materialize with `.collect()` at the end

### Text Namespace Efficiency

- **Expression-level** operations are lazy and benefit from Polars query optimization
- **Series-level** operations are eager and execute immediately
- **Preference**: Use expressions in `select()`/`with_columns()` over series operations

### Auto-Detection Cost

- `guess_document_column()` computes average text length for all string columns
- **Mitigation**: Explicitly specify `document_column` when reading large files
- **Caching**: DocDataFrame stores `_document_column` after detection

## Testing Strategy

### Test Coverage Areas

1. **Namespace Registration**: Verify `.text` accessible on all Polars types
2. **Document Auto-Detection**: Test heuristic with various data patterns
3. **Text Operations**: Unit tests for each namespace method
4. **Serialization**: Round-trip tests for save/load with document metadata
5. **I/O Wrapping**: Test docio decorator with all supported formats
6. **Edge Cases**: Empty strings, None values, non-string columns

### Test Files

- `test_namespace.py`: Namespace registration and method calls
- `test_core.py`: DocDataFrame/DocLazyFrame core functionality
- `test_text_processing.py`: Text operation correctness
- `test_io_comprehensive.py`: I/O decorator behavior
- `test_doclazyframe.py`: Lazy evaluation patterns

## Common Usage Patterns for AI Tools

### Pattern 1: Quick Text Statistics

```python
import docframe as df

# Auto-load and analyze
doc_df = df.read_csv('corpus.csv')
stats = doc_df.describe_text()
print(f"Active document column: {doc_df.active_document_name}")
```

### Pattern 2: Text Preprocessing Pipeline

```python
processed = (
    df.read_csv('raw_data.csv', document_column='content')
    .filter_by_length(min_words=10)
    .clean_documents(lowercase=True, remove_punct=True)
    .with_columns(pl.col('content').text.tokenize().alias('tokens'))
)
```

### Pattern 3: Token Frequency Analysis with Grouping

```python
from docframe import compute_token_frequencies

freq_df = compute_token_frequencies(
    df,
    token_column='tokens',
    group_columns=['category', 'year'],
    method='count',
    normalize=True,
    top_n=50
)
```

### Pattern 4: KWIC Concordance Search

```python
concordances = (
    doc_df
    .select([
        pl.col('id'),
        pl.col('text').text.concordance('climate', num_left_tokens=10, num_right_tokens=10)
    ])
    .explode('text')
    .unnest('text')  # Unpack struct into columns
)
```

### Pattern 5: Document-Term Matrix for ML

```python
# Create DTM
dtm = doc_df.to_dtm(method='tfidf', min_df=2, max_df=0.95)

# Use with sklearn
from sklearn.decomposition import NMF
model = NMF(n_components=10)
topics = model.fit_transform(dtm.to_numpy())
```

## Troubleshooting Guide

### Issue: Text namespace not available

**Cause**: `import docframe` not called  
**Solution**: Import docframe before using `pl.col().text.*`

### Issue: Document column auto-detection fails

**Cause**: No string columns or ambiguous text columns  
**Solution**: Explicitly specify `document_column='column_name'`

### Issue: DTM creation slow on large corpus

**Cause**: Eager tokenization of entire dataset  
**Solution**: Use DocLazyFrame and materialize only DTM result

### Issue: NLTK data not found

**Cause**: First-time NLTK usage without downloads  
**Solution**: Functions auto-download; ensure internet connection or pre-download with `nltk.download('punkt')`

## Future Extension Opportunities

1. **spaCy Integration**: Named entity recognition, dependency parsing
2. **Parallel Processing**: Distribute text operations across cores/nodes
3. **Custom Tokenizers**: Support for non-English languages, domain-specific tokenization
4. **Streaming API**: Process text streams without loading full dataset
5. **Visualization**: Built-in word clouds, concordance plots, topic visualization
6. **Advanced ML**: Integrated embedding generation, semantic similarity

## Conclusion

DocFrame's architecture emphasizes:

- **Composability**: Polars operations + text operations work seamlessly
- **Extensibility**: Namespace pattern allows easy addition of new text methods
- **Performance**: Lazy-by-default with Polars backend for scalability
- **Simplicity**: Auto-detection and intuitive API reduce boilerplate

AI tools can leverage DocFrame by:

- Using `guess_document_column()` for intelligent column detection
- Calling text namespace methods for standardized text processing
- Building on `compute_token_frequencies()` for frequency analysis
- Extending namespace classes for domain-specific operations
