# DocFrame

A powerful text analysis library inspired by GeoPandas design philosophy, using Polars as the backend for efficient document processing and analysis.

## 🚀 Features

- **DocDataFrame & DocLazyFrame**: Document-aware DataFrames with automatic document column detection
- **Polars Backend**: Leverages Polars' performance advantages for large-scale text processing  
- **Text Namespace**: Unified text processing API via Polars namespace registration (`df.text`, `series.text`, `pl.col().text`)
- **Intelligent Auto-Detection**: Automatically identifies document columns using longest average text length heuristic
- **Rich Text Processing**: Built-in tokenization, cleaning, n-grams, word/character/sentence counting
- **Memory Efficient**: Lazy evaluation and optimized memory usage through Polars
- **Comprehensive I/O**: Support for CSV, Parquet, JSON, Excel, and more with document column preservation
- **Serialization**: JSON-based serialization with complete metadata preservation
- **Document Management**: Easy document column switching, renaming, and manipulation

## 📦 Installation

```bash
pip install docframe
```

## 🚀 Quick Start

### Creating DocDataFrames

```python
import docframe as dp

# From dictionary (auto-detects document column)
df = dp.DocDataFrame({
    'title': ['Short title', 'Another title'],
    'content': [
        'This is a much longer document with substantial content for analysis',
        'Another detailed document with comprehensive text for processing'
    ],
    'category': ['news', 'blog']
})

# DocFrame automatically detects 'content' as the document column
print(f"Document column: {df.active_document_name}")  # content

# From list of texts with metadata
df = dp.DocDataFrame.from_texts(
    texts=['Hello world!', 'Text analysis is fun.', 'Polars is fast.'],
    metadata={
        'author': ['Alice', 'Bob', 'Charlie'],
        'category': ['greeting', 'opinion', 'fact']
    }
)
```

### Text Processing

```python
# Access document text directly
documents = df.document  # Returns polars Series

# Add text statistics
df_stats = (df
    .add_word_count()
    .add_char_count() 
    .add_sentence_count()
)

# Text cleaning and processing
df_processed = df.clean_documents(
    lowercase=True,
    remove_punct=True,
    remove_extra_whitespace=True
)

# Filter by text properties
long_docs = df.filter_by_length(min_words=10)
pattern_docs = df.filter_by_pattern(r'\\b(analysis|processing)\\b')

# Get text statistics summary
stats = df.describe_text()
print(stats)
```

### Text Namespace Usage

```python
import polars as pl
import docframe  # Registers text namespace

# Use text namespace on expressions
df_with_tokens = df.select([
    pl.col('*'),
    pl.col('document').text.tokenize().alias('tokens'),
    pl.col('document').text.word_count().alias('word_count'),
    pl.col('document').text.char_count().alias('char_count'),
    pl.col('document').text.clean().alias('cleaned_text')
])

# Advanced text processing
df_advanced = df.select([
    pl.col('*'),
    pl.col('document').text.ngrams(n=2).alias('bigrams'),
    pl.col('document').text.sentence_count().alias('sentences')
])
```

### Document-Term Matrix

```python
# Create document-term matrix for text analysis
dtm = df.to_dtm(method='count')
print(dtm.head())

# Binary DTM
dtm_binary = df.to_dtm(method='binary')

# TF-IDF (requires additional dependencies)
dtm_tfidf = df.to_dtm(method='tfidf')
```

### I/O Operations

```python
# Read files with automatic document column detection
df = dp.read_csv('documents.csv')  # Auto-detects document column
df = dp.read_parquet('data.parquet', document_column='text')
df = dp.read_json('data.json', document_column='content')

# Write preserving DocDataFrame structure
df.write_csv('output.csv')
df.write_parquet('output.parquet')

# Lazy operations for large datasets
lazy_df = dp.scan_csv('large_file.csv')
processed = (lazy_df
    .filter(pl.col('category') == 'news')
    .select([
        pl.col('*'),
        pl.col('document').text.word_count().alias('words')
    ])
    .collect()  # Returns DocDataFrame
)
```

### Data Conversion

```python
# Convert from pandas
import pandas as pd
pdf = pd.DataFrame({'text': ['hello', 'world'], 'label': ['A', 'B']})
df = dp.from_pandas(pdf, document_column='text')

# Convert to regular polars DataFrame
polars_df = df.to_polars()

# Convert to lazy frame
lazy_df = df.to_doclazyframe()
```

### Document Column Management

```python
# Switch document column
df_switched = df.set_document('title')  # Use 'title' as document column

# Rename document column
df_renamed = df.rename_document('text')  # Rename 'document' to 'text'

# Join with document preservation
other_df = pl.DataFrame({'id': [1, 2], 'extra': ['A', 'B']})
joined = df.join(other_df, on='id')  # Preserves DocDataFrame type
```

### Serialization

```python
# Serialize with complete metadata preservation
json_str = df.serialize('json')

# Restore exact DocDataFrame
df_restored = dp.DocDataFrame.deserialize(json_str, format='json')
assert df_restored.active_document_name == df.active_document_name
```

## 🎯 Advanced Examples

### Large-Scale Text Analysis

```python
# Process large document collections efficiently
large_df = (dp.scan_csv('large_corpus.csv')
    .filter(pl.col('language') == 'en')
    .with_columns([
        pl.col('document').text.word_count().alias('word_count'),
        pl.col('document').text.char_count().alias('char_count')
    ])
    .filter(pl.col('word_count') > 50)
    .collect()
)

# Text analysis pipeline
analysis_results = (large_df
    .add_sentence_count()
    .filter_by_length(min_words=100, max_words=1000)
    .sample(n=1000)
    .describe_text()
)
```

### Multi-Document Processing

```python
# Concatenate multiple document collections
news_docs = dp.read_csv('news.csv')
blog_docs = dp.read_csv('blogs.csv') 
academic_docs = dp.read_csv('papers.csv')

all_docs = dp.concat_documents([news_docs, blog_docs, academic_docs])

# Process by category
results = {}
for category in all_docs['category'].unique():
    category_docs = all_docs.filter(pl.col('category') == category)
    results[category] = {
        'count': len(category_docs),
        'avg_length': category_docs.describe_text()['word_count_mean'][0],
        'vocabulary': category_docs.to_dtm().shape[1]
    }
```

### Custom Text Processing

```python
# Combine DocFrame with custom processing
def analyze_sentiment(text: str) -> float:
    # Your sentiment analysis logic
    return 0.5  # placeholder

# Apply custom functions
df_sentiment = df.with_columns([
    pl.col('document').map_elements(analyze_sentiment, return_dtype=pl.Float64).alias('sentiment')
])

# Complex text filtering
complex_filter = (df
    .filter(
        (pl.col('document').text.word_count() > 20) &
        (pl.col('document').text.sentence_count() > 2) &
        (pl.col('category').is_in(['news', 'academic']))
    )
)
```

## 🏗️ Architecture

DocFrame follows GeoPandas' design philosophy adapted for text data:

- **Document Column**: Like GeoPandas' geometry column, DocFrame centers around a designated document column
- **Transparent Operations**: All Polars operations work seamlessly while preserving document metadata
- **Namespace Integration**: Text processing capabilities integrate directly into Polars' expression system
- **Lazy Evaluation**: Full support for Polars' lazy evaluation for memory-efficient processing

## 📚 API Reference

### Core Classes

- **DocDataFrame**: Document-aware DataFrame for eager evaluation
- **DocLazyFrame**: Document-aware LazyFrame for lazy evaluation

### I/O Functions

- `read_csv()`, `read_parquet()`, `read_json()`, `read_excel()` - Read various formats
- `scan_csv()`, `scan_parquet()` - Lazy reading operations
- `from_pandas()`, `from_arrow()` - Convert from other formats

### Utility Functions

- `concat_documents()` - Concatenate DocDataFrames
- `info()` - Library information

### Text Namespace Methods

Available on `pl.col().text`, `series.text`, and `df.text`:

- `tokenize()` - Tokenize text
- `clean()` - Clean text with various options
- `word_count()`, `char_count()`, `sentence_count()` - Count statistics
- `ngrams()` - Extract n-grams
- `contains_pattern()` - Pattern matching

## 🚧 Performance

DocFrame leverages Polars' performance advantages:

- **Memory Efficiency**: Lazy evaluation and zero-copy operations
- **Parallel Processing**: Automatic parallelization of text operations
- **Columnar Storage**: Efficient memory layout for text data
- **Query Optimization**: Polars' query optimizer works with text operations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-org/docframe.git
cd docframe
pip install -e ".[dev]"
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Polars**: For the excellent backend DataFrame library
- **GeoPandas**: For the design philosophy inspiration
- **NLTK/spaCy**: For text processing concepts

## 📞 Support

- **Documentation**: [Full documentation](https://docframe.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/docframe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/docframe/discussions)

---

**DocFrame** - Making text analysis as intuitive as data analysis. 🚀

