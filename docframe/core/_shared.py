"""Shared utilities for DocDataFrame and DocLazyFrame.

This module centralises duplicated logic:
  * Document column guessing
  * Document column validation
  * (De)serialization helpers and metadata handling

Design goals:
  * Keep public surface of DocDataFrame / DocLazyFrame stable for tests.
  * Provide a single implementation path to reduce divergence risk.
"""

from __future__ import annotations

import json
from io import IOBase
from pathlib import Path
from typing import Callable, Literal, Mapping, Optional, Protocol, TypedDict

import polars as pl

# Supported serialization formats. Only 'json' is implemented.
# 'binary' is kept in the Literal for backward reference but will raise NotImplementedError.
SerializationFormat = Literal["json", "binary"]


class DocumentColumnSupport(Protocol):
    """Protocol for objects exposing a document column name."""

    _document_column_name: str | None

    @property
    def document_column(self) -> str | None:  # pragma: no cover - protocol only
        ...


def guess_document_column(
    schema: pl.Schema | Mapping[str, pl.DataType],
    sampler: Callable[[int], pl.DataFrame],
    sample_size: int = 1000,
) -> Optional[str]:
    """Guess the document column.

    Heuristic: choose the string column (Utf8/String) with the largest average
    character length over a sample of rows. Returns None if no string columns.
    """
    items = schema.items() if hasattr(schema, "items") else list(schema)  # type: ignore[arg-type]
    string_columns = [col for col, dtype in items if dtype in (pl.Utf8, pl.String)]
    if not string_columns:
        return None
    if len(string_columns) == 1:
        return string_columns[0]
    sample_df = sampler(sample_size)
    avg_lengths: dict[str, float] = {}
    for col in string_columns:
        avg_length = sample_df.select(pl.col(col).str.len_chars().mean()).item()
        avg_lengths[col] = float(avg_length or 0)
    return max(avg_lengths.keys(), key=lambda k: avg_lengths[k])


def validate_document_column(
    schema: pl.Schema | Mapping[str, pl.DataType], column_name: str
) -> None:
    """Validate existence and string type of a document column."""
    if column_name not in schema:  # type: ignore[operator]
        raise ValueError(f"Document column '{column_name}' not found in data")
    dtype = schema[column_name]  # type: ignore[index]
    if dtype not in (pl.Utf8, pl.String):
        raise ValueError(f"Column '{column_name}' is not a string column")


class _DocMetadata(TypedDict):
    document_column_name: str | None
    type: str


def serialize_with_metadata(
    df: pl.DataFrame,
    document_column: str | None,
    file: IOBase | str | Path | None = None,
    *,
    format: SerializationFormat = "json",
    container_type: str = "DocDataFrame",
) -> str | None:
    """Serialize a DataFrame with document column metadata (JSON only).

    The previous binary format has been removed. Passing format="binary" will
    raise NotImplementedError to make the change explicit to callers.
    """
    if format == "binary":  # explicit signal
        raise NotImplementedError(
            "Binary serialization has been removed; use format='json'"
        )
    if format != "json":  # defensive in case of future literals
        raise ValueError(f"Unsupported format: {format!r}. Only 'json' is supported")

    metadata: _DocMetadata = {
        "document_column_name": document_column,
        "type": container_type,
    }
    df_dict = df.to_dict(as_series=False)
    payload = {"metadata": metadata, "data": df_dict}
    text = json.dumps(payload)
    if file is not None:
        if isinstance(file, (str, Path)):
            with open(file, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            file.write(text)  # type: ignore[arg-type]
        return None
    return text


def deserialize_with_metadata(
    source: str | Path | IOBase | bytes,
    *,
    format: SerializationFormat = "json",
) -> tuple[_DocMetadata, pl.DataFrame]:
    """Deserialize JSON produced by serialize_with_metadata.

    Passing format="binary" raises NotImplementedError (binary removed).
    """
    if format == "binary":
        raise NotImplementedError(
            "Binary deserialization has been removed; use format='json'"
        )
    if format != "json":
        raise ValueError(f"Unsupported format: {format!r}. Only 'json' is supported")

    if isinstance(source, (str, Path)):
        # attempt direct JSON parse; if fails treat as file path
        try:
            text = str(source)
            obj = json.loads(text)
        except json.JSONDecodeError:
            with open(source, "r", encoding="utf-8") as f:
                obj = json.load(f)
    elif isinstance(source, IOBase):
        obj = json.loads(source.read())  # type: ignore[arg-type]
    else:
        raise TypeError("Unsupported source type for json deserialization")

    metadata = obj["metadata"]
    df_dict = obj["data"]
    df = pl.DataFrame(df_dict)
    return metadata, df
