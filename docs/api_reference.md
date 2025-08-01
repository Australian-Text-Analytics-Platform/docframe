# DocFrame API Reference

## Overview

DocFrame is a text analysis library that extends Polars with text processing capabilities, inspired by the GeoPandas design philosophy. It provides document-aware DataFrames and LazyFrames with automatic document column detection and a unified text processing API.

## Core Classes

### DocDataFrame

Document-aware wrapper around Polars DataFrame with automatic document column detection and text processing methods.

```python
from docframe import DocDataFrame

# Create from dictionary (auto-detects document column)
df = DocDataFrame({
    'title': ['Short title', 'Another title'],
    'content': [
        'This is a much longer document with substantial content for analysis',
        'Another detailed document with comprehensive text for processing'
    ],
    'category': ['news', 'blog']
})

# From list of texts with metadata
df = DocDataFrame.from_texts(
    texts=['Hello world!', 'Text analysis is fun.'],
    metadata={'author': ['Alice', 'Bob']}
)
```

#### Properties

##### `document: pl.Series`
Access the active document column as a Polars Series.

##### `active_document_name: str`
Get the name of the currently active document column.

##### `columns: List[str]`
Get all column names in the DataFrame.

#### Core Methods

##### `set_document(column_name: str) -> DocDataFrame`
Switch the active document column to a different column.

##### `rename_document(new_name: str) -> DocDataFrame`
Rename the active document column.

##### `to_polars() -> pl.DataFrame`
Convert to a regular Polars DataFrame.

##### `to_doclazyframe() -> DocLazyFrame`
Convert to a DocLazyFrame for lazy evaluation.

#### Text Analysis Methods

##### `add_word_count(column_name: Optional[str] = None) -> DocDataFrame`
Add a word count column for the specified or active document column.

##### `add_char_count(column_name: Optional[str] = None) -> DocDataFrame`
Add a character count column for the specified or active document column.

##### `add_sentence_count(column_name: Optional[str] = None) -> DocDataFrame`
Add a sentence count column for the specified or active document column.

##### `clean_documents(lowercase=True, remove_punct=True, remove_digits=False, remove_extra_whitespace=True) -> DocDataFrame`
Clean the active document column with various text processing options.

##### `filter_by_length(min_words: Optional[int] = None, max_words: Optional[int] = None) -> DocDataFrame`
Filter rows based on document word count.

##### `filter_by_pattern(pattern: str, case_sensitive: bool = False) -> DocDataFrame`
Filter rows where documents match a regex pattern.

##### `describe_text() -> pl.DataFrame`
Generate comprehensive text statistics for the active document column.

#### Document-Term Matrix

##### `to_dtm(method: str = 'count', min_df: int = 1, max_df: float = 1.0) -> pl.DataFrame`
Create a document-term matrix. Methods: 'count', 'binary', 'tfidf'.

#### Serialization

##### `serialize(filepath: Optional[str] = None, format: str = 'json') -> str`
Serialize the DocDataFrame to JSON format with document column metadata.

##### `deserialize(filepath_or_data: Union[str, dict], format: str = 'json') -> DocDataFrame` (classmethod)
Deserialize from JSON format, restoring document column metadata.

#### Class Methods

##### `guess_document_column(df: pl.DataFrame) -> str` (classmethod)
Automatically detect the best document column using longest average text length heuristic.

---

### DocLazyFrame

Document-aware wrapper around Polars LazyFrame with identical API to DocDataFrame for memory-efficient lazy evaluation.

```python
from docframe import DocLazyFrame
import polars as pl

# Create from LazyFrame
lazy_data = pl.DataFrame({
    'content': ['Hello world!', 'Text analysis is fun.'],
    'author': ['Alice', 'Bob']
}).lazy()

doc_lazy = DocLazyFrame(lazy_data, document_column="content")

# Process with lazy evaluation
result = (doc_lazy
    .add_word_count()
    .filter_by_length(min_words=2)
    .collect()  # Returns DocDataFrame
)
```

#### Properties
Same as DocDataFrame: `document`, `active_document_name`, `columns`

#### Core Methods
Same as DocDataFrame, but operations return DocLazyFrame until `.collect()` is called.

##### `collect() -> DocDataFrame`
Execute the lazy operations and return a DocDataFrame.

##### `to_docdataframe() -> DocDataFrame`
Alias for `collect()`.

#### Text Analysis Methods
Identical to DocDataFrame but with lazy evaluation.

#### Serialization
Same serialization methods as DocDataFrame, returns JSON string only (no file writing).

---

## I/O Functions

### Reading Functions

##### `read_csv(file: str, document_column: Optional[str] = None, **kwargs) -> DocDataFrame`
Read CSV file with automatic document column detection.

##### `read_parquet(file: str, document_column: Optional[str] = None, **kwargs) -> DocDataFrame`
Read Parquet file with automatic document column detection.

##### `read_json(file: str, document_column: Optional[str] = None, **kwargs) -> DocDataFrame`
Read JSON file with automatic document column detection.

##### `read_excel(file: str, document_column: Optional[str] = None, **kwargs) -> DocDataFrame`
Read Excel file with automatic document column detection.

### Lazy Reading Functions

##### `scan_csv(file: str, document_column: Optional[str] = None, **kwargs) -> DocLazyFrame`
Lazily read CSV file with automatic document column detection.

##### `scan_parquet(file: str, document_column: Optional[str] = None, **kwargs) -> DocLazyFrame`
Lazily read Parquet file with automatic document column detection.

### Conversion Functions

##### `from_pandas(df: pd.DataFrame, document_column: Optional[str] = None) -> DocDataFrame`
Convert from Pandas DataFrame with document column detection.

##### `from_arrow(data, document_column: Optional[str] = None) -> DocDataFrame`
Convert from PyArrow Table with document column detection.

### Utility Functions

##### `concat_documents(dfs: List[Union[DocDataFrame, DocLazyFrame]], **kwargs) -> DocDataFrame`
Concatenate multiple DocDataFrames or DocLazyFrames.

##### `info() -> str`
Get library information and version.

---

## Text Namespace

DocFrame registers a `text` namespace with Polars, providing unified text processing across all Polars types.

### Usage Patterns

```python
import polars as pl
import docframe  # Registers text namespace

# Expression namespace (use in select, filter, with_columns)
df.select([
    pl.col('text').text.tokenize().alias('tokens'),
    pl.col('text').text.word_count().alias('word_count'),
    pl.col('text').text.clean().alias('clean_text')
])

# Series namespace (direct operations)
series = pl.Series("texts", ["Hello world!", "Text analysis"])
tokens = series.text.tokenize()
word_counts = series.text.word_count()

# DataFrame namespace (convenience methods)
df_with_stats = df.text.word_count("content").text.char_count("content")
```

### Available Operations

| Method | Description | Return Type |
|--------|-------------|-------------|
| `tokenize()` | Split text into tokens | `pl.Series` (list of strings) |
| `clean(lowercase=True, remove_punct=True, remove_digits=False, remove_extra_whitespace=True)` | Clean text with various options | `pl.Series` (string) |
| `word_count()` | Count words in text | `pl.Series` (integer) |
| `char_count()` | Count characters in text | `pl.Series` (integer) |
| `sentence_count()` | Count sentences in text | `pl.Series` (integer) |
| `ngrams(n=2)` | Extract n-grams | `pl.Series` (list of strings) |
| `contains_pattern(pattern, case_sensitive=False)` | Check if text matches pattern | `pl.Series` (boolean) |
| `remove_stopwords(stopwords=None)` | Remove stopwords from tokenized text | `pl.Series` (list of strings) |

### Expression Namespace

For use in Polars expressions:

```python
# Basic text statistics
df.select([
    pl.col('document').text.word_count().alias('words'),
    pl.col('document').text.char_count().alias('chars'),
    pl.col('document').text.sentence_count().alias('sentences')
])

# Text filtering
df.filter(pl.col('document').text.contains_pattern(r'\b(analysis|research)\b'))

# Text cleaning pipeline
df.with_columns([
    pl.col('document').text.clean().alias('clean_text'),
    pl.col('document').text.tokenize().alias('tokens')
])
```

### Series Namespace

For direct operations on Series:

```python
series = pl.Series("texts", ["Hello world!", "Text analysis is fun."])

# Apply text operations
word_counts = series.text.word_count()
tokens = series.text.tokenize()
clean_text = series.text.clean(lowercase=True, remove_punct=True)

# Chain operations
processed = series.text.clean().text.tokenize()
```

### DataFrame/LazyFrame Namespace

Convenience methods that add new columns:

```python
# Add text statistics columns
df_with_stats = (df
    .text.word_count("document")  # Adds document_word_count
    .text.char_count("document")  # Adds document_char_count
    .text.sentence_count("document")  # Adds document_sentence_count
)

# Works with lazy evaluation
result = (df.lazy()
    .text.clean("document")  # Adds document_clean
    .text.word_count("document")  # Adds document_word_count
    .collect()
)
```

---

## Type System

### Document Column Detection

DocFrame uses an intelligent heuristic to automatically detect document columns:

1. **Identify string columns** in the DataFrame/LazyFrame
2. **Calculate average text length** for each string column (first 1000 rows for performance)
3. **Select column with longest average text length** as the document column
4. **Fall back to 'document'** if no string columns found

### Type Preservation

- **DocDataFrame operations** return DocDataFrame with preserved document column metadata
- **DocLazyFrame operations** return DocLazyFrame until `.collect()` is called
- **Joins and concatenations** preserve DocDataFrame/DocLazyFrame types
- **Text namespace operations** return appropriate Polars types (Series, DataFrame, LazyFrame)

### Serialization Format

JSON serialization preserves complete metadata:

```json
{
    "data": [...],  // Polars DataFrame data
    "document_column": "content",  // Active document column
    "metadata": {
        "library": "docframe",
        "version": "1.0.0",
        "created_at": "2024-01-01T00:00:00Z"
    }
}
```

---

## Error Handling

### Common Exceptions

- **`DocumentColumnError`**: Raised when document column operations fail
- **`SerializationError`**: Raised when serialization/deserialization fails  
- **`TextProcessingError`**: Raised when text processing operations fail

### Best Practices

1. **Always check document column**: Use `df.active_document_name` to verify
2. **Handle missing text**: Text operations handle null values gracefully
3. **Validate input data**: Ensure text columns contain string data
4. **Use try-catch**: Wrap operations in appropriate error handling

```python
try:
    df = DocDataFrame(data)
    print(f"Document column: {df.active_document_name}")
    result = df.add_word_count().filter_by_length(min_words=5)
except Exception as e:
    print(f"Error processing documents: {e}")
```

---

### DocDataFrame

Text-aware wrapper around polars DataFrame with a dedicated 'document' column.

```python
from atapcorpus import DocDataFrame

# Create from texts
texts = ["Hello world", "This is a test"]
metadata = {"author": ["Alice", "Bob"]}
df = DocDataFrame.from_texts(texts, metadata)
```

#### Class Methods

##### `from_texts(texts, metadata=None, document_column='document') -> DocDataFrame`
Create DocDataFrame from list of texts and optional metadata.

##### `from_csv(file_path, document_column='document', **kwargs) -> DocDataFrame`
Read DocDataFrame from CSV file.

##### `guess_document_column(df, sample_size=1000) -> str`
**Class method** to automatically detect the most likely document column in a polars DataFrame.

**Parameters:**
- `df` (pl.DataFrame): The DataFrame to analyze
- `sample_size` (int, default 1000): Number of rows to sample for calculating average length

**Returns:**
- `str` or `None`: Name of the column with longest average string length, or None if no string columns found

**Example:**
```python
import polars as pl
from atapcorpus import DocDataFrame

df = pl.DataFrame({
    'title': ['Short', 'Brief', 'Quick'],
    'content': ['Long document text...', 'Another long text...', 'More content...']
})

best_column = DocDataFrame.guess_document_column(df)
print(best_column)  # 'content'
```

#### Properties

##### `document -> DocSeries`
Access the document column as DocSeries.

##### `document_column -> str`
Name of the document column.

##### `document_column_name -> str`
**Property** that returns the name of the currently designated document column.

**Returns:**
- `str`: Name of the document column

**Example:**
```python
# With auto-detection
df = DocDataFrame({'text': ['doc1', 'doc2'], 'id': [1, 2]})
print(df.document_column_name)  # 'text'

# With manual specification
df = DocDataFrame({'col1': ['a', 'b'], 'col2': ['doc1', 'doc2']}, document_column='col2')
print(df.document_column_name)  # 'col2'
```

##### `dataframe -> pl.DataFrame`
Access underlying polars DataFrame.

#### Text Processing Methods

##### `tokenize(lowercase=True, remove_punct=True) -> pl.Series`
Tokenize all documents.

##### `clean_documents(**kwargs) -> DocDataFrame`
Clean the document column, returns new DocDataFrame.

##### `add_word_count(column_name='word_count') -> DocDataFrame`
Add word count column.

##### `add_char_count(column_name='char_count') -> DocDataFrame`
Add character count column.

##### `add_sentence_count(column_name='sentence_count') -> DocDataFrame`
Add sentence count column.

#### Filtering Methods

##### `filter_by_length(min_words=None, max_words=None) -> DocDataFrame`
Filter documents by word count.

##### `filter_by_pattern(pattern, case_sensitive=False) -> DocDataFrame`
Filter documents containing a pattern.

##### `sample(n=None, fraction=None, seed=None) -> DocDataFrame`
Sample documents.

#### Data Management

##### `add_metadata(metadata) -> DocDataFrame`
Add metadata columns from dictionary.

##### `remove_columns(columns) -> DocDataFrame`
Remove columns (cannot remove document column).

##### `rename_columns(mapping) -> DocDataFrame`
Rename columns using mapping dictionary.

##### `select_columns(columns) -> DocDataFrame`
Select specific columns (automatically includes document column).

#### Export Methods

##### `to_polars() -> pl.DataFrame`
Convert to polars DataFrame.

##### `to_pandas()`
Convert to pandas DataFrame.

##### `to_csv(file_path, **kwargs)`
Write to CSV file.

##### `to_parquet(file_path, **kwargs)`
Write to Parquet file.

##### `to_json(file_path, **kwargs)`
Write to JSON file.

#### Aggregation

##### `groupby_agg(group_by, agg_func='count', text_agg='concat') -> pl.DataFrame`
Group by columns and aggregate.

- `group_by`: Column(s) to group by
- `agg_func`: Aggregation function for numeric columns ('count', 'sum', 'mean', etc.)
- `text_agg`: How to aggregate text ('concat' or 'list')

#### Utility Methods

##### `describe() -> pl.DataFrame`
Generate descriptive statistics including text-specific metrics.

##### `info() -> str`
Display information about the DocDataFrame.

---

## Text Processing Utilities

The `text_utils` module provides standalone text processing functions:

```python
from atapcorpus.core.text_utils import (
    simple_tokenize, clean_text, word_count, 
    char_count, sentence_count, extract_ngrams,
    contains_pattern, remove_stopwords
)

# Tokenize text
tokens = simple_tokenize("Hello world!", lowercase=True, remove_punct=True)

# Clean text
clean = clean_text("Hello World!", lowercase=True, remove_punct=True)

# Count words
count = word_count("Hello world")

# Extract bigrams
bigrams = extract_ngrams("hello world test", n=2)

# Check pattern
has_pattern = contains_pattern("hello world", "world")

# Remove stopwords
filtered = remove_stopwords(["the", "quick", "brown", "fox"])
```

---

## Polars Integration

ATAPCorpus is designed to work seamlessly with polars:

```python
import polars as pl
from atapcorpus import DocDataFrame

# Create DocDataFrame
df = DocDataFrame.from_texts(["Hello world", "Test document"])

# Access underlying polars DataFrame
polars_df = df.dataframe

# Use polars operations
result = polars_df.with_columns([
    pl.col("document").str.len_chars().alias("char_length"),
    pl.col("document").str.to_uppercase().alias("upper_text")
])

# Convert back to DocDataFrame
new_df = DocDataFrame(result)
```

---

## Performance Tips

1. **Batch Operations**: Use DocDataFrame methods for batch processing rather than iterating.

2. **Lazy Evaluation**: Consider using polars lazy API for complex pipelines:
   ```python
   lazy_result = df.dataframe.lazy().with_columns([
       pl.col("document").map_elements(lambda x: len(x.split())).alias("words")
   ]).collect()
   ```

3. **Memory Efficiency**: Polars handles memory efficiently, but for very large datasets consider:
   - Processing in chunks
   - Using lazy evaluation
   - Selecting only needed columns

4. **Custom Operations**: For complex text processing, use polars expressions:
   ```python
   df.dataframe.with_columns([
       pl.col("document").str.extract_all(r'\w+').alias("words"),
       pl.col("document").str.count_matches(r'[A-Z]').alias("capitals")
   ])
   ```

---

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Introduction to core functionality
- `advanced_usage.py`: Complex text analysis pipelines
