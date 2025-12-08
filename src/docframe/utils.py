"""
DocFrame Utilities - Common functions for text data analysis
Similar to GeoPandas utilities for working with geographic data
"""

import warnings
import zipfile
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List

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


_DEFAULT_TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".text",
    ".md",
    ".rst",
    ".log",
    ".cfg",
    ".ini",
    ".yml",
    ".yaml",
    ".json",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".tex",
    ".srt",
}


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
            except (ValueError, AssertionError) as e:
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


def _read_zip(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    errors: str = "ignore",
    text_extensions: Iterable[str] | None = None,
    include_extensionless: bool = True,
) -> pl.DataFrame:
    """Read text files from a ZIP archive into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the ZIP archive.
    encoding : str, default "utf-8"
        Text encoding used when decoding file contents.
    errors : str, default "ignore"
        Error handling strategy passed to :py:meth:`bytes.decode`.
    text_extensions : Iterable[str], optional
        Custom iterable of file extensions that should be treated as text.
        Extensions can be specified with or without a leading dot and are
        matched case-insensitively.
    include_extensionless : bool, default True
        Whether files without an extension should be treated as text.

    Returns
    -------
    polars.DataFrame
        DataFrame with columns ``file_path``, ``base_name``, ``extension`` and ``document``.
    """

    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"ZIP archive not found: {archive_path}")

    if text_extensions is None:
        allowed_extensions = _DEFAULT_TEXT_FILE_EXTENSIONS
    else:
        allowed_extensions = {
            (ext if ext.startswith(".") else f".{ext}").lower()
            for ext in text_extensions
        }

    records: list[dict[str, str]] = []

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            file_path = info.filename
            path_obj = Path(file_path)
            file_name = path_obj.name
            base_name = path_obj.stem
            extension = path_obj.suffix

            # Skip macOS resource forks and hidden system entries
            if file_path.startswith("__MACOSX/") or file_name.startswith("._"):
                continue

            suffix = extension.lower()
            if suffix:
                if suffix not in allowed_extensions:
                    continue
            elif not include_extensionless:
                continue

            with archive.open(info, "r") as file_obj:
                data = file_obj.read()

            try:
                text_content = data.decode(encoding, errors=errors)
            except UnicodeDecodeError:
                warnings.warn(
                    f"Skipping '{file_path}' - unable to decode with encoding {encoding!r}",
                    UserWarning,
                )
                continue

            records.append({
                "file_path": file_path,
                "base_name": base_name,
                "extension": extension,
                "document": text_content,
            })

    records.sort(key=lambda entry: entry["file_path"])

    return pl.DataFrame(
        records,
        schema={
            "file_path": pl.String,
            "base_name": pl.String,
            "extension": pl.String,
            "document": pl.String,
        },
    )


# Conditionally import and wrap functions that may not exist in all polars versions
def _ensure_fastexcel_available() -> None:
    try:
        import fastexcel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DocFrame Excel support requires the 'fastexcel' package. "
            "Install it via 'uv add fastexcel' or 'pip install fastexcel'."
        ) from exc


try:
    from polars import read_excel as _read_excel
except ImportError:  # pragma: no cover - dependent on polars build
    _read_excel = None
else:

    def _docframe_read_excel(*args, **kwargs):
        _ensure_fastexcel_available()
        kwargs.pop("engine", None)
        try:
            return _read_excel(*args, **kwargs)
        except ModuleNotFoundError as exc:  # pragma: no cover - fastexcel missing
            raise ImportError(
                "Polars could not import its default Excel engine bindings. "
                "Ensure the 'fastexcel' package is installed."
            ) from exc

    read_excel = docio(_docframe_read_excel)

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
_read_zip_wrapped = docio(_read_zip)


def _read_text_file(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    errors: str = "ignore",
) -> pl.DataFrame:
    """Read a plain-text file into a single-column DataFrame."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    text_content = file_path.read_text(encoding=encoding, errors=errors)

    return pl.DataFrame({"document": [text_content]}, schema={"document": pl.String})


_read_text_wrapped = docio(_read_text_file)


def read_zip(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    errors: str = "ignore",
    text_extensions: Iterable[str] | None = None,
    include_extensionless: bool = True,
    document_column: str | None = "document",
):
    """Read textual members from a ZIP archive into a DocDataFrame."""

    return _read_zip_wrapped(
        path,
        encoding=encoding,
        errors=errors,
        text_extensions=text_extensions,
        include_extensionless=include_extensionless,
        document_column=document_column,
    )


def read_text(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    errors: str = "ignore",
    document_column: str | None = "document",
):
    """Read a single plain-text document into a DocDataFrame."""

    return _read_text_wrapped(
        path,
        encoding=encoding,
        errors=errors,
        document_column=document_column,
    )


def excel_sheet_names(path: str | Path) -> List[str]:
    """Return the available worksheet names for an Excel workbook."""

    if _read_excel is None:  # pragma: no cover - depends on polars build
        raise ImportError(
            "This version of Polars does not ship with read_excel support"
        )

    _ensure_fastexcel_available()
    try:
        sheets = _read_excel(path, sheet_id=None)
    except ModuleNotFoundError as exc:  # pragma: no cover - fastexcel missing
        raise ImportError(
            "Polars could not import its default Excel engine bindings. Ensure fastexcel is installed."
        ) from exc

    if isinstance(sheets, dict):
        return list(sheets.keys())
    if isinstance(sheets, pl.DataFrame):
        # Polars returns a DataFrame when sheet_id=None on single-sheet workbooks
        return ["Sheet1"]
    return []


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
