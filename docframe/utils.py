"""
DocFrame Utilities - Common functions for text data analysis
Similar to GeoPandas utilities for working with geographic data
"""

import warnings
from functools import wraps
from typing import TYPE_CHECKING, Callable, List

import polars as pl
from polars import from_arrow as _from_arrow
from polars import from_numpy as _from_numpy
from polars import from_pandas as _from_pandas
from polars import read_csv as _read_csv
from polars import read_json as _read_json
from polars import read_ndjson as _read_ndjson
from polars import read_parquet as _read_parquet
from polars import scan_csv as _scan_csv
from polars import scan_ndjson as _scan_ndjson
from polars import scan_parquet as _scan_parquet

from .core.docframe import DocDataFrame

if TYPE_CHECKING:
    pass


def docio(func: Callable) -> Callable:
    """
    Decorator that adds document_column support to any polars I/O function.

    This decorator wraps polars I/O functions to automatically convert results
    to DocDataFrame when document_column is explicitly provided.

    Behavior:
    - When document_column is not provided or None: auto-detects best document column using guess_document_column()
    - When document_column='column_name': uses specified column as document column
    - When document_column=False: disables conversion, always returns regular polars objects
    - Returns DocDataFrame when successful, regular polars objects when auto-detection fails or errors occur
    - Issues warnings for invalid column specifications but gracefully falls back to regular objects

    Parameters
    ----------
    func : Callable
        The polars I/O function to wrap

    Returns
    -------
    Callable
        Wrapped function with document_column parameter

    Examples
    --------
    >>> from polars import read_csv as pl_read_csv
    >>> read_csv = docio(pl_read_csv)
    >>>
    >>> # Auto-detects document column, returns DocDataFrame if successful
    >>> doc_df = read_csv('data.csv')
    >>>
    >>> # Explicitly triggers auto-detection
    >>> doc_df = read_csv('data.csv', document_column=None)
    >>>
    >>> # Uses specified document column
    >>> doc_df = read_csv('data.csv', document_column='text')
    >>>
    >>> # Disables conversion, returns regular DataFrame
    >>> df = read_csv('data.csv', document_column=False)
    """

    @wraps(func)
    def wrapper(
        *args, **kwargs
    ) -> DocDataFrame | pl.DataFrame | pl.LazyFrame | pl.Series:
        # Get document_column parameter, defaulting to None for auto-detection
        document_column = kwargs.pop("document_column", None)

        # Call the original polars function
        result = func(*args, **kwargs)

        # Always try to convert to DocDataFrame/DocLazyFrame for DataFrame/LazyFrame unless explicitly disabled with False
        if document_column is not False and isinstance(
            result, pl.DataFrame | pl.LazyFrame
        ):
            # If document_column is None, use guess_document_column
            document_column = document_column or DocDataFrame.guess_document_column(
                result
            )

            try:
                if isinstance(result, pl.LazyFrame):
                    return result.text.to_doclazyframe(document_column=document_column)
                else:
                    return result.text.to_docdataframe(document_column=document_column)
            except ValueError as e:
                warnings.warn(
                    f"Could not create DocDataFrame/DocLazyFrame: {e}", UserWarning
                )
                return result

        # For Series, just return the series as-is (users can use .text namespace directly)
        return result

    return wrapper


# Apply the decorator to create enhanced versions
read_csv = docio(_read_csv)
read_parquet = docio(_read_parquet)
read_json = docio(_read_json)
read_ndjson = docio(_read_ndjson)
scan_csv = docio(_scan_csv)
scan_parquet = docio(_scan_parquet)
scan_ndjson = docio(_scan_ndjson)
from_pandas = docio(_from_pandas)
from_arrow = docio(_from_arrow)
from_numpy = docio(_from_numpy)

# Conditionally import and wrap functions that may not exist in all polars versions
try:
    from polars import read_excel as _read_excel

    read_excel = docio(_read_excel)
except ImportError:
    pass

try:
    from polars import read_database as _read_database

    read_database = docio(_read_database)
except ImportError:
    pass

try:
    from polars import read_ipc as _read_ipc

    read_ipc = docio(_read_ipc)
except ImportError:
    pass

try:
    from polars import read_avro as _read_avro

    read_avro = docio(_read_avro)
except ImportError:
    pass

try:
    from polars import read_delta as _read_delta

    read_delta = docio(_read_delta)
except ImportError:
    pass


# Import and wrap polars I/O functions using the decorator


def concat_documents(
    doc_dfs: List[DocDataFrame], how: str = "vertical"
) -> DocDataFrame:
    """
    Concatenate multiple DocDataFrames.

    Parameters
    ----------
    doc_dfs : list of DocDataFrame
        List of DocDataFrames to concatenate
    how : str, default "vertical"
        How to concatenate ("vertical" or "horizontal")

    Returns
    -------
    DocDataFrame
        Concatenated DocDataFrame

    Raises
    ------
    ValueError
        If DocDataFrames have different document column names
    """
    if not doc_dfs:
        raise ValueError("No DocDataFrames provided")

    # Check if all are DocDataFrame
    if not all(isinstance(df, DocDataFrame) for df in doc_dfs):
        raise ValueError("All items must be DocDataFrame")

    # DocDataFrame concatenation
    doc_col_name = doc_dfs[0].active_document_name
    for df in doc_dfs[1:]:
        if df.active_document_name != doc_col_name:
            raise ValueError(
                "All DocDataFrames must have the same document column name"
            )

    # Concatenate underlying DataFrames
    pl_dfs = [df._df for df in doc_dfs]

    if how == "vertical":
        result_df = pl.concat(pl_dfs, how="vertical")
    elif how == "horizontal":
        result_df = pl.concat(pl_dfs, how="horizontal")
    else:
        raise ValueError("how must be 'vertical' or 'horizontal'")

    return DocDataFrame(result_df, document_column=doc_col_name)


def info() -> str:
    """
    Return information about DocFrame.

    Returns
    -------
    str
        Information about the library
    """
    return """
DocFrame - Text Analysis with Polars
=====================================

A GeoPandas-inspired library for text analysis built on polars.

Key Features:
• DocDataFrame: Text-aware DataFrame with dedicated document column
• Text namespace: All text processing via series.text.method() pattern
• Automatic document column detection
• High-performance text processing
• Polars namespace integration
• GeoPandas-like API design
• Smart decorator-based I/O with document_column support

Quick Start:
>>> import docframe
>>> df = docframe.read_csv('data.csv', document_column='text')
>>> df.document.text.tokenize()  # Text processing via namespace
>>> df.add_word_count().filter_by_length(min_words=10)

I/O Functions with document_column support:
>>> doc_df = docframe.read_csv('file.csv', document_column='text')
>>> doc_df = docframe.read_parquet('file.parquet', document_column='auto')
>>> doc_df = docframe.from_pandas(pandas_df, document_column='content')

For more information, see the documentation.
    """.strip()
