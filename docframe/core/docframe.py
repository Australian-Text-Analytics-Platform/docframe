"""
DocDataFrame - Document-aware polars DataFrame for LDaCA
"""

import json
from io import IOBase
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl


class DocDataFrame:
    """
    A document-aware wrapper around polars DataFrame for LDaCA with a dedicated 'document' column.

    DocDataFrame extends polars.DataFrame with a special 'document' column for document analysis.
    """

    @classmethod
    def guess_document_column(
        cls, df: pl.DataFrame | pl.LazyFrame, sample_size: int = 1000
    ) -> Optional[str]:
        """
        Guess the best document column by finding the string column with the longest average length.

        Parameters
        ----------
        df : pl.DataFrame | pl.LazyFrame
            The DataFrame or LazyFrame to analyze
        sample_size : int, default 1000
            Number of rows to sample for calculating average length

        Returns
        -------
        str or None
            Name of the column with longest average string length, or None if no string columns found
        """
        # Get string columns - use collect_schema() for LazyFrame to avoid performance warning
        if isinstance(df, pl.LazyFrame):
            schema = df.collect_schema()
        else:
            schema = df.schema
        # Accept both Utf8 and String aliases from Polars
        string_columns = [
            col for col, dtype in schema.items() if dtype in (pl.Utf8, pl.String)
        ]

        if not string_columns:
            return None

        if len(string_columns) == 1:
            return string_columns[0]

        # Calculate average length for each string column (using first sample_size rows)
        # Handle both DataFrame and LazyFrame
        if isinstance(df, pl.LazyFrame):
            sample_df = df.head(sample_size).collect()
        else:
            sample_df = df.head(min(sample_size, len(df)))

        avg_lengths = {}

        for col in string_columns:
            # Calculate average length, handling nulls
            avg_length = sample_df.select(pl.col(col).str.len_chars().mean()).item()
            avg_lengths[col] = avg_length or 0  # Handle None case

        # Return column with longest average length
        return max(avg_lengths.keys(), key=lambda k: avg_lengths[k])

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

        # Ensure document column exists
        if self._document_column_name not in columns:
            if isinstance(data, dict) and self._document_column_name in data:
                pass  # Column will be created from dict
            else:
                raise ValueError(
                    f"Document column '{self._document_column_name}' not found in data"
                )

        # Validate that document column is a string type (if column exists)
        if self._document_column_name in columns:
            column_type = schema[self._document_column_name]
            if column_type not in (pl.Utf8, pl.String):
                raise ValueError(
                    f"Column '{self._document_column_name}' is not a string column"
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
        """Tokenize documents"""
        return self.document.text.tokenize(
            lowercase=lowercase, remove_punct=remove_punct
        )

    def clean_documents(
        self,
        lowercase: bool = True,
        remove_punct: bool = True,
        remove_digits: bool = False,
        remove_extra_whitespace: bool = True,
    ) -> "DocDataFrame":
        """Clean the document column"""
        cleaned_docs = self.document.text.clean(
            lowercase=lowercase,
            remove_punct=remove_punct,
            remove_digits=remove_digits,
            remove_extra_whitespace=remove_extra_whitespace,
        )

        # Replace the document column with cleaned version
        result_df = self._df.with_columns(
            cleaned_docs.alias(self._document_column_name)
        )
        return DocDataFrame(result_df, document_column=self._document_column_name)

    def add_word_count(self, column_name: str = "word_count") -> "DocDataFrame":
        """Add word count column"""
        word_counts = self.document.text.word_count()
        result_df = self._df.with_columns(word_counts.alias(column_name))
        return DocDataFrame(result_df, document_column=self._document_column_name)

    def add_char_count(self, column_name: str = "char_count") -> "DocDataFrame":
        """Add character count column"""
        char_counts = self.document.text.char_count()
        result_df = self._df.with_columns(char_counts.alias(column_name))
        return DocDataFrame(result_df, document_column=self._document_column_name)

    def add_sentence_count(self, column_name: str = "sentence_count") -> "DocDataFrame":
        """Add sentence count column"""
        sentence_counts = self.document.text.sentence_count()
        result_df = self._df.with_columns(sentence_counts.alias(column_name))
        return DocDataFrame(result_df, document_column=self._document_column_name)

    def filter_by_length(
        self, min_words: Optional[int] = None, max_words: Optional[int] = None
    ) -> "DocDataFrame":
        """Filter documents by word count"""
        word_counts = self.document.text.word_count()

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
        """Filter documents containing a pattern"""
        mask = self.document.text.contains_pattern(
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
        sparse_matrix, vocabulary = self.document.text.to_dtm(method=method, **kwargs)

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
            other_df = other._df
        else:
            other_df = other

        joined_df = self._df.join(other_df, *args, **kwargs)
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
                    doc_series.text.word_count().mean(),
                    doc_series.text.word_count().std(),
                    doc_series.text.char_count().mean(),
                    doc_series.text.char_count().std(),
                ],
            }
        )

        return text_stats

    # Serialization methods
    def serialize(
        self,
        file: IOBase | str | Path | None = None,
        *,
        format: str = "json",
    ) -> str | None:
        """
        Serialize DocDataFrame to JSON string or file, preserving document column information.

        Parameters
        ----------
        file : IOBase, str, Path, or None
            File path or file-like object to write to. If None, returns serialized data.
        format : str, default 'json'
            Serialization format. Only 'json' is supported.

        Returns
        -------
        str or None
            Serialized JSON string if file is None, otherwise None
        """
        if format != "json":
            raise ValueError(f"Unsupported format: {format}. Only 'json' is supported")

        metadata = {
            "document_column_name": self._document_column_name,
        }

        # Convert to dictionary format
        df_dict = self._df.to_dict(as_series=False)
        serialized_data = {"metadata": metadata, "data": df_dict}
        result = json.dumps(serialized_data)

        # Handle file output
        if file is not None:
            if isinstance(file, (str, Path)):
                with open(file, "w") as f:
                    f.write(result)
            else:
                # file-like object
                file.write(result)
            return None

        return result

    @classmethod
    def deserialize(
        cls,
        source: str | Path | IOBase,
        *,
        format: str = "json",
    ) -> "DocDataFrame":
        """
        Deserialize JSON data back to DocDataFrame.

        Parameters
        ----------
        source : str, Path, or IOBase
            Source to deserialize from. Can be a file path, file-like object, or JSON string.
        format : str, default "json"
            Serialization format. Only 'json' is supported.

        Returns
        -------
        DocDataFrame
            Deserialized DocDataFrame
        """
        if format != "json":
            raise ValueError(f"Unsupported format: {format}. Only 'json' is supported")

        # Read data from source
        if isinstance(source, (str, Path)):
            # Check if it's a file path or JSON string
            try:
                # Try to parse as JSON string first
                serialized_data = json.loads(str(source))
                data = str(source)
            except json.JSONDecodeError:
                # It's a file path, read from file
                with open(source, "r") as f:
                    data = f.read()
        else:
            # file-like object
            data = source.read()

        # Parse JSON data
        serialized_data = json.loads(data)
        metadata = serialized_data["metadata"]
        df_dict = serialized_data["data"]

        # Reconstruct DataFrame
        df = pl.DataFrame(df_dict)

        return cls(df, document_column=metadata["document_column_name"])

    # Delegate other operations to underlying DataFrame
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying polars DataFrame"""
        attr = getattr(self._df, name)

        # If it's a method that returns a DataFrame, wrap it in DocDataFrame
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # If result is a DataFrame and contains our document column, wrap it
                if (
                    isinstance(result, pl.DataFrame)
                    and self._document_column_name in result.columns
                ):
                    return DocDataFrame(
                        result, document_column=self._document_column_name
                    )
                return result

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
        """
        Guess the best document column by finding the string column with the longest average length.

        For LazyFrames, this uses schema information and samples data for analysis.

        Parameters
        ----------
        df : pl.DataFrame | pl.LazyFrame
            The DataFrame or LazyFrame to analyze
        sample_size : int, default 1000
            Number of rows to sample for calculating average length

        Returns
        -------
        str or None
            Name of the column with longest average string length, or None if no string columns found
        """
        # Get string columns - use collect_schema() for LazyFrame to avoid performance warning
        if isinstance(df, pl.LazyFrame):
            schema = df.collect_schema()
        else:
            schema = df.schema
        # Accept both Utf8 and String aliases from Polars
        string_columns = [
            col for col, dtype in schema.items() if dtype in (pl.Utf8, pl.String)
        ]

        if not string_columns:
            return None

        if len(string_columns) == 1:
            return string_columns[0]

        # Calculate average length for each string column (using first sample_size rows)
        # Handle both DataFrame and LazyFrame
        if isinstance(df, pl.LazyFrame):
            sample_df = df.head(sample_size).collect()
        else:
            sample_df = df.head(min(sample_size, len(df)))

        avg_lengths = {}

        for col in string_columns:
            # Calculate average length, handling nulls
            avg_length = sample_df.select(pl.col(col).str.len_chars().mean()).item()
            avg_lengths[col] = avg_length or 0  # Handle None case

        # Return column with longest average length
        return max(avg_lengths.keys(), key=lambda k: avg_lengths[k])

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
            # Validate the column exists
            schema = self._df.collect_schema()
            if document_column not in schema:
                raise ValueError(
                    f"Document column '{document_column}' not found in LazyFrame"
                )

            # Validate that document column is a string type
            column_type = schema[document_column]
            if column_type not in (pl.Utf8, pl.String):
                raise ValueError(f"Column '{document_column}' is not a string column")

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

    def serialize(self, format: str = "json") -> str:
        """
        Serialize the DocLazyFrame to JSON.

        Parameters
        ----------
        format : str, default "json"
            Serialization format. Only 'json' is supported.

        Returns
        -------
        str
            Serialized JSON string
        """
        if format != "json":
            raise ValueError(f"Unsupported format: {format}. Only 'json' is supported")

        # For JSON, we need to collect first
        collected = self.collect()
        json_data = collected.serialize(format="json")
        serialized_data = {
            "type": "DocLazyFrame",
            "data": json.loads(json_data),
            "document_column": self._document_column_name,
        }
        return json.dumps(serialized_data)

    @classmethod
    def deserialize(
        cls, source: str | Path | IOBase, format: str = "json"
    ) -> "DocLazyFrame":
        """
        Deserialize JSON data back to DocLazyFrame.

        Parameters
        ----------
        source : str, Path, or IOBase
            Source to deserialize from. Can be a file path, file-like object, or JSON string.
        format : str, default "json"
            Serialization format. Only 'json' is supported.

        Returns
        -------
        DocLazyFrame
            Deserialized DocLazyFrame
        """
        if format != "json":
            raise ValueError(f"Unsupported format: {format}. Only 'json' is supported")

        # Read data from source
        if isinstance(source, (str, Path)):
            # Check if it's a file path or JSON string
            try:
                # Try to parse as JSON string first
                data = json.loads(str(source))
            except json.JSONDecodeError:
                # It's a file path, read from file
                with open(source, "r") as f:
                    data = json.loads(f.read())
        else:
            # file-like object
            data = json.loads(source.read())

        if data["type"] != "DocLazyFrame":
            raise ValueError(f"Expected DocLazyFrame data, got {data['type']}")

        # Deserialize the inner DocDataFrame and convert to LazyFrame
        doc_data = data["data"]
        doc_df = DocDataFrame.deserialize(json.dumps(doc_data))
        lazyframe = doc_df._df.lazy()

        return cls(lazyframe, document_column=data["document_column"])

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

                # Wrap LazyFrame results as DocLazyFrame
                if isinstance(result, pl.LazyFrame):
                    # Check if document column still exists
                    try:
                        schema = result.collect_schema()
                        if (
                            self._document_column_name
                            and self._document_column_name in schema
                        ):
                            return DocLazyFrame(
                                result, document_column=self._document_column_name
                            )
                    except Exception:
                        pass
                    return DocLazyFrame(result)

                # Wrap DataFrame results as DocDataFrame
                elif isinstance(result, pl.DataFrame):
                    if (
                        self._document_column_name
                        and self._document_column_name in result.columns
                    ):
                        return DocDataFrame(
                            result, document_column=self._document_column_name
                        )
                    return DocDataFrame(result)

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
