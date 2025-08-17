"""
DocDataFrame - Document-aware polars DataFrame for LDaCA
"""

import json
from io import BytesIO, IOBase
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import polars as pl

SerializationFormat = Literal["json", "binary"]


def guess_document_column(
    schema: pl.Schema,
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


class _DocumentColumnMixin:
    """Shared helpers for document column aware objects (eager & lazy).

    Provides a unified implementation of guess_document_column used by
    both DocDataFrame and DocLazyFrame to avoid duplication.
    """

    @classmethod
    def guess_document_column(
        cls, df: pl.DataFrame | pl.LazyFrame, sample_size: int = 1000
    ) -> Optional[str]:
        if isinstance(df, pl.LazyFrame):
            schema = df.collect_schema()

            def sampler(n: int):
                return df.head(n).collect()

        else:
            schema = df.schema

            def sampler(n: int):
                return df.head(min(n, len(df)))

        return guess_document_column(schema, sampler, sample_size=sample_size)

    def validate_document_eligibility(self, column_name: str) -> None:
        """Validate if the given column is eligible to be a document column."""
        if column_name not in self.columns:
            raise ValueError(f"Column '{column_name}' is not a valid document column")

        schema = (
            self.schema if isinstance(self, pl.DataFrame) else self.collect_schema()
        )
        if schema[column_name] not in (pl.Utf8, pl.String):
            raise ValueError(f"Column '{column_name}' is not a string column")

    # Shared properties -------------------------------------------------
    @property
    def document_column(self) -> Optional[str]:
        """Name of the active document column (None if not set)."""
        return getattr(self, "_document_column_name", None)

    @property
    def active_document_name(self) -> Optional[str]:
        """Alias maintained for backward compatibility."""
        return self.document_column

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}, document column: {self.document_column}\n({repr(self._df)})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}, document column: {self.document_column}\n({str(self._df)})"

    # Shared item access ------------------------------------------------
    def __getitem__(self, key):  # type: ignore[override]
        """Delegate subscription to underlying object and wrap if appropriate.

        Works for both eager (DataFrame) and lazy (LazyFrame) wrappers.
        """
        result = self._df[key]  # type: ignore[attr-defined]
        doc_col = getattr(self, "_document_column_name", None)

        if doc_col:
            try:
                if isinstance(result, pl.DataFrame):
                    if doc_col in result.columns:
                        return DocDataFrame(result, document_column=doc_col)
                elif isinstance(result, pl.LazyFrame):
                    if doc_col in result.collect_schema().names():
                        return DocLazyFrame(result, document_column=doc_col)
            except Exception:
                return result
        return result

    def set_document(self, column_name: str):
        """Return a new instance with a different active document column.

        Performs presence & type validation before constructing new wrapper.
        Works for both eager (DataFrame) and lazy (LazyFrame) variants.
        """
        # Determine schema depending on underlying frame type
        df_obj = self._df  # type: ignore[attr-defined]
        self.validate_document_eligibility(column_name)
        return self.__class__(df_obj, document_column=column_name)  # type: ignore[misc]

    # Shared attribute delegation ---------------------------------------
    def __getattr__(self, name):  # type: ignore[override]
        """Delegate unknown attributes to underlying polars object.

        Wrap resulting DataFrame/LazyFrame preserving document column when it
        still exists after the operation; otherwise return raw result.
        """
        # Avoid recursion: only called if normal lookup fails.
        df_obj = getattr(self, "_df", None)
        if df_obj is None:
            raise AttributeError(name)

        # Special-case virtual 'columns' to avoid repeated schema collection for lazy frames
        if name == "columns":
            if isinstance(df_obj, pl.LazyFrame):
                return df_obj.collect_schema().names()
            return df_obj.columns  # type: ignore[return-value]

        attr = getattr(df_obj, name)
        if not callable(attr):
            return attr

        doc_col = getattr(self, "_document_column_name", None)

        def wrapper(*args, **kwargs):
            result = attr(*args, **kwargs)
            if isinstance(result, pl.DataFrame):
                if doc_col and doc_col in result.columns:
                    return DocDataFrame(result, document_column=doc_col)
                return result
            if isinstance(result, pl.LazyFrame):
                try:
                    if doc_col and doc_col in result.collect_schema().names():
                        return DocLazyFrame(result, document_column=doc_col)
                except Exception:
                    # If schema collection fails, return raw result
                    return result
                return result
            return result

        return wrapper

    # Serialization (shared) -------------------------------------------
    def serialize(
        self,
        file: IOBase | str | Path | None = None,
        *,
        format: SerializationFormat = "json",
    ) -> str | None:
        """Serialize object (JSON only).

        DocDataFrame: direct metadata + DataFrame data (via shared helper).
        DocLazyFrame: collects then wraps with type marker to preserve prior shape.
        """
        if format == "binary":
            raise NotImplementedError("Binary serialization removed; use format='json'")
        if format != "json":
            raise ValueError(
                f"Unsupported format: {format!r}. Only 'json' is supported"
            )

        doc_col = getattr(self, "_document_column_name", None)
        if doc_col is None:
            raise ValueError("No document column set for serialization")

        # Unified serialization: always {'metadata': {...}, 'data': ...}
        data_json = self._df.serialize(format="json")  # type: ignore[attr-defined]
        if isinstance(self, DocDataFrame):
            type_label = "DocDataFrame"
        elif isinstance(self, DocLazyFrame):
            type_label = "DocLazyFrame"
        else:  # pragma: no cover
            raise TypeError("Unsupported instance type for serialization")

        metadata = {"document_column_name": doc_col, "type": type_label}
        payload = {"metadata": metadata, "data": data_json}
        text = json.dumps(payload)
        if file is not None:
            if isinstance(file, (str, Path)):
                with open(file, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                file.write(text)  # type: ignore[arg-type]
            return None
        return text

        # Fallback (should not happen)
        raise TypeError("Unsupported instance type for serialization")

    @classmethod
    def deserialize(
        cls,
        source: str | Path | IOBase | bytes,
        *,
        format: SerializationFormat = "json",
    ):
        """Polymorphic deserialization based on cls target.

        For DocDataFrame: delegate to shared helper.
        For DocLazyFrame: expect wrapper with type field.
        """
        if format == "binary":
            raise NotImplementedError(
                "Binary deserialization removed; use format='json'"
            )
        if format != "json":
            raise ValueError(
                f"Unsupported format: {format!r}. Only 'json' is supported"
            )

        # Load object (string may be direct JSON or file path)
        if isinstance(source, (str, Path)):
            try:
                obj = json.loads(str(source))
            except json.JSONDecodeError:
                with open(source, "r", encoding="utf-8") as f:
                    obj = json.load(f)
        else:
            obj = json.loads(source.read())

        if not isinstance(obj, dict) or "metadata" not in obj or "data" not in obj:
            raise ValueError("Invalid serialized object: missing 'metadata' or 'data'")

        metadata = obj["metadata"]
        doc_col = metadata.get("document_column_name")
        type_label = metadata.get("type")

        data_part = obj["data"]
        if cls is DocDataFrame:
            if type_label != "DocDataFrame":
                raise ValueError(
                    f"Unexpected type label {type_label!r} for DocDataFrame"
                )
            df = pl.DataFrame.deserialize(BytesIO(data_part.encode()), format="json")

        elif cls is DocLazyFrame:
            if type_label != "DocLazyFrame":
                raise ValueError(
                    f"Unexpected type label {type_label!r} for DocLazyFrame"
                )
            df = pl.LazyFrame.deserialize(BytesIO(data_part.encode()), format="json")
        else:
            raise TypeError(f"Deserialization for {cls.__name__} not supported")

        return cls(df, document_column=doc_col)

        # Text-specific methods that operate on the document column

    def tokenize(self, lowercase: bool = True, remove_punct: bool = True) -> pl.Series:
        """Tokenize documents."""
        return self._df.tokenize(
            column=self.document_column, lowercase=lowercase, remove_punct=remove_punct
        )


class DocDataFrame(_DocumentColumnMixin):
    """
    A document-aware wrapper around polars DataFrame for LDaCA with a dedicated 'document' column.

    DocDataFrame extends polars.DataFrame with a special 'document' column for document analysis.
    """

    # guess_document_column inherited from _DocumentColumnMixin

    def __init__(
        self,
        data: Union[pl.DataFrame, Dict[str, Any], None] = None,
        document_column: Optional[str] = None,
    ):
        """
        Initialize DocDataFrame

        Parameters
        ----------
        data : pl.DataFrame, dict, or None
            The data to initialize with. Only DataFrames are supported.
        document_column : str, optional
            Name of the column containing documents. If None, will try to auto-detect
            the string column with the longest average length, or default to 'document'
        """
        if data is None:
            data = {}

        if isinstance(data, pl.DataFrame):
            self._df = data
        elif isinstance(data, dict):
            self._df = pl.DataFrame(data)
        else:
            raise ValueError("data must be a polars DataFrame or dictionary")

        # Auto-detect document column if not specified
        if document_column is None:
            guessed_column = self.guess_document_column(self._df)
            if guessed_column is not None:
                document_column = guessed_column
            else:
                document_column = "document"  # fallback default

        # Set the document column name (like geometry_column_name in GeoPandas)
        self._document_column_name = document_column

        # Validate column existence + type
        self.validate_document_eligibility(self._document_column_name)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        metadata: Optional[Dict[str, List[Any]]] = None,
        document_column: str = "document",
    ) -> "DocDataFrame":
        """
        Create DocDataFrame from list of texts

        Parameters
        ----------
        texts : list of str
            List of text documents
        metadata : dict, optional
            Dictionary of metadata columns
        document_column : str, default 'document'
            Name for the document column

        Returns
        -------
        DocDataFrame
            New DocDataFrame instance
        """
        data = {document_column: texts}

        if metadata:
            for key, values in metadata.items():
                if len(values) != len(texts):
                    raise ValueError(
                        f"Metadata column '{key}' length {len(values)} "
                        f"doesn't match texts length {len(texts)}"
                    )
                data[key] = values

        return cls(data, document_column=document_column)

    @property
    def dataframe(self) -> pl.DataFrame:
        """Access underlying polars DataFrame"""
        return self._df

    @property
    def document(self) -> pl.Series:
        """Access the document column as polars Series with text processing capabilities"""
        return self._df[self._document_column_name]

    def rename_document(self, new_name: str) -> "DocDataFrame":
        """
        Rename the document column.

        Parameters
        ----------
        new_name : str
            New name for the document column

        Returns
        -------
        DocDataFrame
            New DocDataFrame with renamed document column
        """
        # Get columns
        columns = self._df.columns

        if new_name in columns and new_name != self._document_column_name:
            raise ValueError(f"Column '{new_name}' already exists")

        renamed_df = self._df.rename({self._document_column_name: new_name})
        return DocDataFrame(renamed_df, document_column=new_name)

    def __len__(self) -> int:
        return len(self._df)

    def clean_documents(
        self,
        lowercase: bool = True,
        remove_punct: bool = True,
        remove_digits: bool = False,
        remove_extra_whitespace: bool = True,
    ) -> "DocDataFrame":
        """Return a new DocDataFrame with cleaned document text."""
        cleaned_docs = self.document.text.clean(  # type: ignore[attr-defined]
            lowercase=lowercase,
            remove_punct=remove_punct,
            remove_digits=remove_digits,
            remove_extra_whitespace=remove_extra_whitespace,
        )
        result_df = self._df.with_columns(
            cleaned_docs.alias(self._document_column_name)
        )
        return DocDataFrame(result_df, document_column=self._document_column_name)

    def add_word_count(self, column_name: str = "word_count") -> "DocDataFrame":
        word_counts = self.document.text.word_count()  # type: ignore[attr-defined]
        result_df = self._df.with_columns(word_counts.alias(column_name))
        return DocDataFrame(result_df, document_column=self._document_column_name)

    def add_char_count(self, column_name: str = "char_count") -> "DocDataFrame":
        char_counts = self.document.text.char_count()  # type: ignore[attr-defined]
        result_df = self._df.with_columns(char_counts.alias(column_name))
        return DocDataFrame(result_df, document_column=self._document_column_name)

    def add_sentence_count(self, column_name: str = "sentence_count") -> "DocDataFrame":
        sentence_counts = self.document.text.sentence_count()  # type: ignore[attr-defined]
        result_df = self._df.with_columns(sentence_counts.alias(column_name))
        return DocDataFrame(result_df, document_column=self._document_column_name)

    def filter_by_length(
        self, min_words: Optional[int] = None, max_words: Optional[int] = None
    ) -> "DocDataFrame":
        word_counts = self.document.text.word_count()  # type: ignore[attr-defined]
        if min_words is not None and max_words is not None:
            mask = (word_counts >= min_words) & (word_counts <= max_words)
        elif min_words is not None:
            mask = word_counts >= min_words
        elif max_words is not None:
            mask = word_counts <= max_words
        else:
            mask = pl.Series([True] * len(word_counts))
        filtered_df = self._df.filter(mask)
        return DocDataFrame(filtered_df, document_column=self._document_column_name)

    def filter_by_pattern(
        self, pattern: str, case_sensitive: bool = False
    ) -> "DocDataFrame":
        mask = self.document.text.contains_pattern(  # type: ignore[attr-defined]
            pattern, case_sensitive=case_sensitive
        )
        filtered_df = self._df.filter(mask)
        return DocDataFrame(filtered_df, document_column=self._document_column_name)

    # Data export methods
    def to_dataframe(self) -> pl.DataFrame:
        """Convert to polars DataFrame"""
        return self._df

    def to_doclazyframe(self) -> "DocLazyFrame":
        """
        Convert to DocLazyFrame for lazy evaluation.

        Returns
        -------
        DocLazyFrame
            New DocLazyFrame with the same data and document column
        """
        return DocLazyFrame(self._df.lazy(), document_column=self._document_column_name)

    def to_dtm(self, method: str = "count", **kwargs):
        """
        Create Document-Term Matrix from the document column.

        Parameters
        ----------
        method : str, default "count"
            Method for DTM creation. Options: "count", "tfidf", "binary"
        **kwargs
            Additional arguments passed to sklearn vectorizer

        Returns
        -------
        tuple[scipy.sparse matrix, list[str]]
            Sparse DTM matrix and feature names (vocabulary)

        Examples
        --------
        >>> dtm_df = doc_df.to_dtm(method="tfidf", max_features=1000)
        >>> dtm_df = doc_df.to_dtm(method="count", min_df=2, max_df=0.8)
        """
        # Get sparse matrix and vocabulary from the text namespace
        sparse_matrix, vocabulary = self.document.text.to_dtm(  # type: ignore[attr-defined]
            method=method, **kwargs
        )

        # Convert sparse matrix to dense and create DataFrame
        dense_matrix = sparse_matrix.toarray()
        dtm_data = {
            vocab_word: dense_matrix[:, i] for i, vocab_word in enumerate(vocabulary)
        }

        return pl.DataFrame(dtm_data)

    def join(
        self, other: Union["DocDataFrame", pl.DataFrame, pl.LazyFrame], *args, **kwargs
    ) -> "DocDataFrame":
        """
        Join with another DocDataFrame or polars DataFrame.

        Parameters
        ----------
        other : DocDataFrame, pl.DataFrame, or pl.LazyFrame
            DataFrame to join with
        *args, **kwargs
            Additional arguments passed to polars join method

        Returns
        -------
        DocDataFrame
            New DocDataFrame with joined data
        """
        if isinstance(other, DocDataFrame):
            other_df: pl.DataFrame = other._df
        elif isinstance(other, pl.LazyFrame):
            other_df = other.collect()
        else:
            other_df = other  # type: ignore[assignment]

        joined_df = self._df.join(other_df, *args, **kwargs)  # type: ignore[arg-type]
        return DocDataFrame(joined_df, document_column=self._document_column_name)

class DocLazyFrame(_DocumentColumnMixin):
    """
    A text-aware wrapper around polars LazyFrame with a dedicated 'document' column.

    This provides lazy evaluation capabilities while maintaining text analysis functionality.
    Similar to DocDataFrame but for lazy operations.
    """

    # guess_document_column inherited from _DocumentColumnMixin

    def __init__(
        self,
        data: pl.LazyFrame,
        document_column: Optional[str] = None,
    ):
        """
        Initialize a DocLazyFrame.

        Parameters
        ----------
        data : pl.LazyFrame
            The underlying polars LazyFrame
        document_column : str, optional
            Name of the document column. If None, will attempt to guess.
        """
        if not isinstance(data, pl.LazyFrame):
            raise TypeError(f"Expected pl.LazyFrame, got {type(data)}")

        self._df = data

        # Determine document column
        if document_column is None:
            self._document_column_name = self.guess_document_column(self._df)
        else:
            self.validate_document_eligibility(document_column)
            self._document_column_name = document_column

    @property
    def lazyframe(self) -> pl.LazyFrame:
        """Access the underlying polars LazyFrame."""
        return self._df

    @property
    def document(self) -> pl.Expr:
        """Get an expression for the document column."""
        if self._document_column_name is None:
            raise ValueError("No document column available")
        return pl.col(self._document_column_name)

    def collect(self) -> "DocDataFrame":
        """
        Collect the LazyFrame into a DocDataFrame.

        Returns
        -------
        DocDataFrame
            The materialized DocDataFrame
        """
        collected_df = self._df.collect()
        return DocDataFrame(collected_df, document_column=self._document_column_name)

    def to_docdataframe(self) -> "DocDataFrame":
        """
        Convert to DocDataframe by collecting the LazyFrame.

        This is an alias for collect() for consistency with to_doclazyframe().

        Returns
        -------
        DocDataFrame
            The materialized DocDataFrame
        """
        return self.collect()

    def to_lazyframe(self) -> pl.LazyFrame:
        """
        Convert to polars LazyFrame (unwrap the underlying LazyFrame).

        Returns
        -------
        pl.LazyFrame
            The underlying polars LazyFrame
        """
        return self._df

    def with_document_column(self, column_name: str) -> "DocLazyFrame":
        """Alias to set_document for backward compatibility."""
        return self.set_document(column_name)  # type: ignore[return-value]
