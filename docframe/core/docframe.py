"""
DocDataFrame - Document-aware polars DataFrame for LDaCA
"""

import json
from io import IOBase
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from ._shared import (
    SerializationFormat,
    deserialize_with_metadata,
    serialize_with_metadata,
)
from ._shared import guess_document_column as _guess_document_column_helper
from ._shared import validate_document_column as _validate_document_column


class DocDataFrame:
    """
    A document-aware wrapper around polars DataFrame for LDaCA with a dedicated 'document' column.

    DocDataFrame extends polars.DataFrame with a special 'document' column for document analysis.
    """

    @classmethod
    def guess_document_column(
        cls, df: pl.DataFrame | pl.LazyFrame, sample_size: int = 1000
    ) -> Optional[str]:
        """Delegates to shared helper for column inference."""
        if isinstance(df, pl.LazyFrame):
            schema = df.collect_schema()

            def sampler(n: int):
                return df.head(n).collect()
        else:
            schema = df.schema

            def sampler(n: int):
                return df.head(min(n, len(df)))

        return _guess_document_column_helper(schema, sampler, sample_size=sample_size)

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

        # Get schema and columns
        schema = self._df.schema
        columns = self._df.columns

        # Validate column existence + type
        if self._document_column_name in columns:
            _validate_document_column(schema, self._document_column_name)
        else:
            raise ValueError(
                f"Document column '{self._document_column_name}' not found in data"
            )

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

    @property
    def active_document_name(self) -> str:
        """Return the name of the active document column."""
        return self._document_column_name

    @property
    def document_column(self) -> str:
        """Get the name of the document column (alias for active_document_name)."""
        return self._document_column_name

    def set_document(self, column_name: str) -> "DocDataFrame":
        """
        Set a different column as the document column.

        Parameters
        ----------
        column_name : str
            Name of the column to set as the document column

        Returns
        -------
        DocDataFrame
            New DocDataFrame with updated document column

        Raises
        ------
        ValueError
            If the specified column doesn't exist or is not a string column
        """
        # Get columns and schema
        schema = self._df.schema
        columns = self._df.columns

        if column_name not in columns:
            raise ValueError(f"Document column '{column_name}' not found")

        # Check if it's a string column
        if schema[column_name] != pl.Utf8:
            raise ValueError(f"Column '{column_name}' is not a string column")

        return DocDataFrame(self._df, document_column=column_name)

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

    def __getitem__(self, key):
        """Access columns or filter rows via delegation to underlying DataFrame"""
        result = self._df[key]

        # If result is a DataFrame and contains our document column, wrap it
        if (
            isinstance(result, pl.DataFrame)
            and self._document_column_name in result.columns
        ):
            return DocDataFrame(result, document_column=self._document_column_name)

        # Otherwise return the raw result (Series, values, etc.)
        return result

    def __repr__(self) -> str:
        doc_info = f", document_column='{self._document_column_name}'"
        return f"DocDataFrame({repr(self._df)}{doc_info})"

    def __str__(self) -> str:
        doc_info = f"Document column: '{self._document_column_name}'\n"
        return doc_info + str(self._df)

    # Text-specific methods that operate on the document column
    def tokenize(self, lowercase: bool = True, remove_punct: bool = True) -> pl.Series:
        """Tokenize documents."""
        return self.document.text.tokenize(  # type: ignore[attr-defined]
            lowercase=lowercase, remove_punct=remove_punct
        )

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

    def sample(
        self,
        n: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "DocDataFrame":
        """Sample documents"""
        if n is not None:
            sampled_df = self._df.sample(n=n, seed=seed)
        elif fraction is not None:
            sampled_df = self._df.sample(fraction=fraction, seed=seed)
        else:
            raise ValueError("Either n or fraction must be specified")

        return DocDataFrame(sampled_df, document_column=self._document_column_name)

    # Data export methods
    def to_polars(self) -> pl.DataFrame:
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

    # Summary methods
    def describe_text(self) -> pl.DataFrame:
        """Generate text-specific descriptive statistics"""
        # Add text-specific metrics for document column
        doc_series = self.document
        text_stats = pl.DataFrame(
            {
                "statistic": [
                    "word_count_mean",
                    "word_count_std",
                    "char_count_mean",
                    "char_count_std",
                ],
                self._document_column_name: [
                    doc_series.text.word_count().mean(),  # type: ignore[attr-defined]
                    doc_series.text.word_count().std(),  # type: ignore[attr-defined]
                    doc_series.text.char_count().mean(),  # type: ignore[attr-defined]
                    doc_series.text.char_count().std(),  # type: ignore[attr-defined]
                ],
            }
        )

        return text_stats

    # Serialization methods
    def serialize(
        self,
        file: IOBase | str | Path | None = None,
        *,
        format: SerializationFormat = "json",
    ) -> str | None:
        """Serialize (JSON only).

        The previous custom binary container has been removed. Passing
        format="binary" raises NotImplementedError to make migration explicit.
        """
        if format == "binary":
            raise NotImplementedError("Binary serialization removed; use format='json'")
        if format != "json":
            raise ValueError(
                f"Unsupported format: {format!r}. Only 'json' is supported"
            )
        return serialize_with_metadata(
            self._df,
            self._document_column_name,
            file=file,
            format="json",
            container_type="DocDataFrame",
        )

    @classmethod
    def deserialize(
        cls,
        source: str | Path | IOBase | bytes,
        *,
        format: SerializationFormat = "json",
    ) -> "DocDataFrame":
        if format == "binary":
            raise NotImplementedError(
                "Binary deserialization removed; use format='json'"
            )
        if format != "json":
            raise ValueError(
                f"Unsupported format: {format!r}. Only 'json' is supported"
            )
        metadata, df = deserialize_with_metadata(source, format="json")
        return cls(df, document_column=metadata["document_column_name"])

    # Delegate other operations to underlying DataFrame
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying polars DataFrame"""
        attr = getattr(self._df, name)

        # If it's a method that returns a DataFrame, wrap it in DocDataFrame
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # New behavior: only wrap if original document column still present.
                # If it was dropped, return raw DataFrame so downstream code can explicitly
                # choose to re-wrap (e.g., DocDataFrame(result) with guessed column) if desired.
                if isinstance(result, pl.DataFrame):
                    if self._document_column_name in result.columns:
                        return DocDataFrame(
                            result, document_column=self._document_column_name
                        )
                    # Document column absent -> return plain DataFrame
                    return result
                return result  # Non-DataFrame results unchanged

            return wrapper

        return attr


class DocLazyFrame:
    """
    A text-aware wrapper around polars LazyFrame with a dedicated 'document' column.

    This provides lazy evaluation capabilities while maintaining text analysis functionality.
    Similar to DocDataFrame but for lazy operations.
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

        return _guess_document_column_helper(schema, sampler, sample_size=sample_size)

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
            schema = self._df.collect_schema()
            _validate_document_column(schema, document_column)
            self._document_column_name = document_column

    @property
    def lazyframe(self) -> pl.LazyFrame:
        """Access the underlying polars LazyFrame."""
        return self._df

    @property
    def document_column(self) -> Optional[str]:
        """Get the name of the document column."""
        return self._document_column_name

    @property
    def active_document_name(self) -> Optional[str]:
        """Get the active document column name (alias for compatibility)."""
        return self._document_column_name

    @property
    def columns(self) -> list[str]:
        """Get column names without triggering schema resolution warning."""
        return self._df.collect_schema().names()

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
        """
        Create a new DocLazyFrame with a different document column.

        Parameters
        ----------
        column_name : str
            Name of the new document column

        Returns
        -------
        DocLazyFrame
            New DocLazyFrame with updated document column
        """
        return DocLazyFrame(self._df, document_column=column_name)

    def serialize(
        self,
        file: IOBase | str | Path | None = None,
        *,
        format: SerializationFormat = "json",
    ) -> str | None:
        """Serialize the collected LazyFrame with metadata (JSON only)."""
        if format == "binary":
            raise NotImplementedError("Binary serialization removed; use format='json'")
        if format != "json":
            raise ValueError(
                f"Unsupported format: {format!r}. Only 'json' is supported"
            )
        collected = self.collect()
        inner = collected.serialize(format="json")
        if inner is None:
            raise ValueError("Unexpected None from inner JSON serialization")
        serialized_data = {
            "type": "DocLazyFrame",
            "data": json.loads(inner),
            "document_column": self._document_column_name,
        }
        text = json.dumps(serialized_data)
        if file is not None:
            if isinstance(file, (str, Path)):
                with open(file, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                file.write(text)  # type: ignore[arg-type]
            return None
        return text

    @classmethod
    def deserialize(
        cls,
        source: str | Path | IOBase | bytes,
        *,
        format: SerializationFormat = "json",
    ) -> "DocLazyFrame":
        if format == "binary":
            raise NotImplementedError(
                "Binary deserialization removed; use format='json'"
            )
        if format != "json":
            raise ValueError(
                f"Unsupported format: {format!r}. Only 'json' is supported"
            )
        if isinstance(source, (str, Path)):
            try:
                obj = json.loads(str(source))
            except json.JSONDecodeError:
                with open(source, "r", encoding="utf-8") as f:
                    obj = json.load(f)
        else:
            obj = json.loads(source.read())  # type: ignore[arg-type]
        if obj.get("type") != "DocLazyFrame":
            raise ValueError(f"Expected DocLazyFrame data, got {obj.get('type')!r}")
        inner = obj["data"]
        metadata_inner = inner["metadata"]["document_column_name"]
        df_inner = pl.DataFrame(inner["data"])
        return cls(df_inner.lazy(), document_column=metadata_inner)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying LazyFrame.

        Operations that return LazyFrames will be wrapped as DocLazyFrames.
        Operations that return DataFrames will be wrapped as DocDataFrames.
        """
        # Special handling for columns to avoid performance warning
        if name == "columns":
            return self.columns

        # Use try/except instead of hasattr to avoid triggering columns access
        try:
            attr = getattr(self._df, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)

                # New behavior: only wrap when original document column persists.
                if isinstance(result, pl.LazyFrame):
                    try:
                        if self._document_column_name is not None:
                            schema = result.collect_schema()
                            if self._document_column_name in schema:
                                return DocLazyFrame(
                                    result, document_column=self._document_column_name
                                )
                    except Exception:
                        pass
                    # Document column not present -> return raw LazyFrame
                    return result

                if isinstance(result, pl.DataFrame):
                    if (
                        self._document_column_name is not None
                        and self._document_column_name in result.columns
                    ):
                        return DocDataFrame(
                            result, document_column=self._document_column_name
                        )
                    # Document column not present -> return raw DataFrame
                    return result

                return result

            return wrapper

        return attr

    def __repr__(self) -> str:
        """String representation."""
        doc_col = self._document_column_name or "None"
        return f"DocLazyFrame(document_column='{doc_col}', lazyframe={repr(self._df)})"

    def __str__(self) -> str:
        doc_info = f"Document column: '{self._document_column_name}'\n"
        return doc_info + str(self._df)
