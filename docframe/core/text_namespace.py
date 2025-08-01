"""
Document processing namespace for polars using official namespace registration - LDaCA
"""

from functools import partial
from typing import List, Optional

import polars as pl

from .docframe import DocDataFrame, DocLazyFrame
from .text_utils import (
    char_count,
    clean_text,
    extract_ngrams,
    remove_stopwords,
    sentence_count,
    simple_tokenize,
    word_count,
)


@pl.api.register_expr_namespace("text")
class TextExprNamespace:
    """Text processing namespace for polars expressions"""

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def tokenize(self, lowercase: bool = True, remove_punct: bool = True) -> pl.Expr:
        """Tokenize text into list of tokens"""

        _tokenize = partial(
            simple_tokenize, lowercase=lowercase, remove_punct=remove_punct
        )

        return self._expr.map_elements(_tokenize, return_dtype=pl.List(pl.String))

    def clean(
        self,
        lowercase: bool = True,
        remove_punct: bool = True,
        remove_digits: bool = False,
        remove_extra_whitespace: bool = True,
    ) -> pl.Expr:
        """Clean text with various options"""

        _clean = partial(
            clean_text,
            lowercase=lowercase,
            remove_punct=remove_punct,
            remove_digits=remove_digits,
            remove_extra_whitespace=remove_extra_whitespace,
        )

        return self._expr.map_elements(_clean, return_dtype=pl.String)

    def word_count(self) -> pl.Expr:
        """Count words in text"""
        _word_count = partial(word_count)

        return self._expr.map_elements(_word_count, return_dtype=pl.Int32)

    def char_count(self) -> pl.Expr:
        """Count characters in text"""

        _char_count = partial(char_count)

        return self._expr.map_elements(_char_count, return_dtype=pl.Int32)

    def sentence_count(self) -> pl.Expr:
        """Count sentences in text"""

        _sentence_count = partial(sentence_count)

        return self._expr.map_elements(_sentence_count, return_dtype=pl.Int32)

    def ngrams(self, n: int = 2) -> pl.Expr:
        """Extract n-grams from text"""

        def _ngrams(text: str) -> List[str]:
            from .text_utils import extract_ngrams

            return extract_ngrams(text, n=n)

        return self._expr.map_elements(_ngrams, return_dtype=pl.List(pl.String))

    def contains_pattern(self, pattern: str, case_sensitive: bool = False) -> pl.Expr:
        """Check if text contains a pattern"""

        def _contains(text: str) -> bool:
            from .text_utils import contains_pattern

            return contains_pattern(text, pattern, case_sensitive=case_sensitive)

        return self._expr.map_elements(_contains, return_dtype=pl.Boolean)

    def remove_stopwords(self, stopwords: Optional[List[str]] = None) -> pl.Expr:
        """Remove stopwords from tokenized text"""

        _remove_stopwords = partial(remove_stopwords, stopwords=stopwords)

        return self._expr.map_elements(
            _remove_stopwords, return_dtype=pl.List(pl.String)
        )

    def join_tokens(self, separator: str = " ") -> pl.Expr:
        """Join list of tokens back into text"""
        return self._expr.list.join(separator)

    def filter_tokens(self, min_length: int = 1) -> pl.Expr:
        """Filter tokens by minimum length"""
        return self._expr.list.eval(
            pl.element().filter(pl.element().str.len_chars() >= min_length)
        )

    def to_dtm(self, method: str = "count", **kwargs):
        """
        Create Document-Term Matrix from text column.
        This method is intended to be used on Series level.
        For DataFrame-level DTM creation, use DocDataFrame.to_dtm()
        """
        raise NotImplementedError(
            "DTM creation from expression level is not supported. "
            "Use Series.text.to_dtm() or DocDataFrame.to_dtm() instead."
        )


@pl.api.register_series_namespace("text")
class TextSeriesNamespace:
    """Text processing namespace for polars Series"""

    def __init__(self, series: pl.Series):
        self._series = series

    def tokenize(self, lowercase: bool = True, remove_punct: bool = True) -> pl.Series:
        """Tokenize text into list of tokens"""
        return (
            self._series.to_frame()
            .select(
                pl.col(self._series.name).text.tokenize(
                    lowercase=lowercase, remove_punct=remove_punct
                )
            )
            .to_series()
        )
        # _tokenize = partial(
        #     simple_tokenize, lowercase=lowercase, remove_punct=remove_punct
        # )

        # return self._series.map_elements(_tokenize, return_dtype=pl.List(pl.String))

    def clean(
        self,
        lowercase: bool = True,
        remove_punct: bool = True,
        remove_digits: bool = False,
        remove_extra_whitespace: bool = True,
    ) -> pl.Series:
        """Clean text with various options"""
        return (
            self._series.to_frame()
            .select(
                pl.col(self._series.name).text.clean(
                    lowercase=lowercase,
                    remove_punct=remove_punct,
                    remove_digits=remove_digits,
                    remove_extra_whitespace=remove_extra_whitespace,
                )
            )
            .to_series()
        )

    def word_count(self) -> pl.Series:
        """Count words in text"""
        return (
            self._series.to_frame()
            .select(pl.col(self._series.name).text.word_count())
            .to_series()
        )

    def char_count(self) -> pl.Series:
        """Count characters in text"""
        return (
            self._series.to_frame()
            .select(pl.col(self._series.name).text.char_count())
            .to_series()
        )

    def sentence_count(self) -> pl.Series:
        """Count sentences in text"""
        return (
            self._series.to_frame()
            .select(pl.col(self._series.name).text.sentence_count())
            .to_series()
        )

    def ngrams(self, n: int = 2) -> pl.Series:
        """Extract n-grams from text"""
        return (
            self._series.to_frame()
            .select(pl.col(self._series.name).text.ngrams(n=n))
            .to_series()
        )

    def contains_pattern(self, pattern: str, case_sensitive: bool = False) -> pl.Series:
        """Check if text contains a pattern"""
        return (
            self._series.to_frame()
            .select(
                pl.col(self._series.name).text.contains_pattern(
                    pattern, case_sensitive=case_sensitive
                )
            )
            .to_series()
        )

    def remove_stopwords(self, stopwords: Optional[List[str]] = None) -> pl.Series:
        """Remove stopwords from tokenized text"""
        return (
            self._series.to_frame()
            .select(
                pl.col(self._series.name).text.remove_stopwords(stopwords=stopwords)
            )
            .to_series()
        )

    def join_tokens(self, separator: str = " ") -> pl.Series:
        """Join list of tokens back into text"""
        return (
            self._series.to_frame()
            .select(pl.col(self._series.name).text.join_tokens(separator=separator))
            .to_series()
        )

    def filter_tokens(self, min_length: int = 1) -> pl.Series:
        """Filter tokens by minimum length"""
        return (
            self._series.to_frame()
            .select(pl.col(self._series.name).text.filter_tokens(min_length=min_length))
            .to_series()
        )

    def to_dtm(self, method: str = "count", **kwargs):
        """
        Create Document-Term Matrix from text series.

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
        >>> dtm, vocab = series.text.to_dtm(method="tfidf", max_features=1000)
        """
        try:
            from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        except ImportError:
            raise ImportError(
                "scikit-learn is required for DTM functionality. Install with: pip install scikit-learn"
            )

        # Convert series to list of documents
        documents = self._series.to_list()

        # Remove None/null values
        documents = [doc for doc in documents if doc is not None]

        if not documents:
            raise ValueError("No valid documents found for DTM creation")

        # Choose vectorizer based on method
        if method == "count":
            vectorizer = CountVectorizer(**kwargs)
        elif method == "tfidf":
            vectorizer = TfidfVectorizer(**kwargs)
        elif method == "binary":
            vectorizer = CountVectorizer(binary=True, **kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Options: 'count', 'tfidf', 'binary'"
            )

        # Create DTM
        dtm = vectorizer.fit_transform(documents)
        vocabulary = vectorizer.get_feature_names_out()

        return dtm, vocabulary.tolist()


@pl.api.register_dataframe_namespace("text")
class TextDataFrameNamespace:
    """Text processing namespace for polars DataFrame"""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def tokenize(
        self, column: str, lowercase: bool = True, remove_punct: bool = True
    ) -> pl.DataFrame:
        """Tokenize text column into list of tokens"""
        return self._df.with_columns(
            pl.col(column)
            .text.tokenize(lowercase=lowercase, remove_punct=remove_punct)
            .alias(f"{column}_tokens")
        )

    def clean(
        self,
        column: str,
        lowercase: bool = True,
        remove_punct: bool = True,
        remove_digits: bool = False,
        remove_extra_whitespace: bool = True,
    ) -> pl.DataFrame:
        """Clean text column with various options"""
        return self._df.with_columns(
            pl.col(column)
            .text.clean(
                lowercase=lowercase,
                remove_punct=remove_punct,
                remove_digits=remove_digits,
                remove_extra_whitespace=remove_extra_whitespace,
            )
            .alias(f"{column}_clean")
        )

    def word_count(self, column: str) -> pl.DataFrame:
        """Count words in text column"""
        return self._df.with_columns(
            pl.col(column).text.word_count().alias(f"{column}_word_count")
        )

    def char_count(self, column: str) -> pl.DataFrame:
        """Count characters in text column"""
        return self._df.with_columns(
            pl.col(column).text.char_count().alias(f"{column}_char_count")
        )

    def sentence_count(self, column: str) -> pl.DataFrame:
        """Count sentences in text column"""
        return self._df.with_columns(
            pl.col(column).text.sentence_count().alias(f"{column}_sentence_count")
        )

    def ngrams(self, column: str, n: int = 2) -> pl.DataFrame:
        """Extract n-grams from text column"""
        return self._df.with_columns(
            pl.col(column).text.ngrams(n=n).alias(f"{column}_ngrams")
        )

    def contains_pattern(
        self, column: str, pattern: str, case_sensitive: bool = False
    ) -> pl.DataFrame:
        """Check if text column contains a pattern"""
        return self._df.with_columns(
            pl.col(column)
            .text.contains_pattern(pattern, case_sensitive=case_sensitive)
            .alias(f"{column}_contains")
        )

    def concordance(
        self,
        column: str,
        search_word: str,
        num_left_tokens: int = 10,
        num_right_tokens: int = 10,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> pl.DataFrame:
        """
        Generate a concordance table for a search word/pattern in a text column.

        Parameters
        ----------
        column : str
            Name of the text column to search in
        search_word : str
            Word or pattern to search for
        num_left_tokens : int, default 10
            Number of tokens to include in left context
        num_right_tokens : int, default 10
            Number of tokens to include in right context
        regex : bool, default False
            Whether to treat search_word as a regex pattern
        case_sensitive : bool, default False
            Whether the search should be case sensitive

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: document_idx, left_context, matched_text, right_context, l1, l1_freq, r1, r1_freq
        """
        import re
        from collections import Counter

        from .text_utils import simple_tokenize

        if len(search_word) == 0:
            return pl.DataFrame(
                {
                    "document_idx": [],
                    "left_context": [],
                    "matched_text": [],
                    "right_context": [],
                    "l1": [],
                    "l1_freq": [],
                    "r1": [],
                    "r1_freq": [],
                },
                schema={
                    "document_idx": pl.Int32,
                    "left_context": pl.String,
                    "matched_text": pl.String,
                    "right_context": pl.String,
                    "l1": pl.String,
                    "l1_freq": pl.Int32,
                    "r1": pl.String,
                    "r1_freq": pl.Int32,
                },
            )

        # Create regex pattern
        pattern = search_word if regex else re.escape(search_word)
        flags = 0 if case_sensitive else re.IGNORECASE
        searcher = re.compile(pattern, flags)

        # Get the text column as a list
        texts = self._df[column].to_list()

        conc_table = []
        l1_tokens = []  # Collect all L1 tokens for frequency calculation
        r1_tokens = []  # Collect all R1 tokens for frequency calculation

        # First pass: collect all matches and extract L1/R1 tokens
        for idx, doc in enumerate(texts):
            if doc is None:
                continue

            for match in searcher.finditer(doc):
                matched_text = match.group(0)
                left_text = doc[: match.start()]
                right_text = doc[match.end() :]

                # Tokenize left and right contexts
                left_tokens = simple_tokenize(
                    left_text, lowercase=False, remove_punct=False
                )
                right_tokens = simple_tokenize(
                    right_text, lowercase=False, remove_punct=False
                )

                # Get the specified number of context tokens
                left_context_tokens = (
                    left_tokens[-num_left_tokens:] if num_left_tokens > 0 else []
                )
                right_context_tokens = (
                    right_tokens[:num_right_tokens] if num_right_tokens > 0 else []
                )

                # Extract L1 and R1 tokens (first left and first right)
                l1 = left_context_tokens[-1] if left_context_tokens else ""
                r1 = right_context_tokens[0] if right_context_tokens else ""

                conc_table.append(
                    {
                        "document_idx": idx,
                        "left_context": " ".join(left_context_tokens),
                        "matched_text": matched_text,
                        "right_context": " ".join(right_context_tokens),
                        "l1": l1,
                        "r1": r1,
                    }
                )

                # Collect for frequency calculation
                if l1:
                    l1_tokens.append(l1)
                if r1:
                    r1_tokens.append(r1)

        if len(conc_table) == 0:
            return pl.DataFrame(
                {
                    "document_idx": [],
                    "left_context": [],
                    "matched_text": [],
                    "right_context": [],
                    "l1": [],
                    "l1_freq": [],
                    "r1": [],
                    "r1_freq": [],
                },
                schema={
                    "document_idx": pl.Int32,
                    "left_context": pl.String,
                    "matched_text": pl.String,
                    "right_context": pl.String,
                    "l1": pl.String,
                    "l1_freq": pl.Int32,
                    "r1": pl.String,
                    "r1_freq": pl.Int32,
                },
            )

        # Calculate frequencies
        l1_freq_counter = Counter(l1_tokens)
        r1_freq_counter = Counter(r1_tokens)

        # Second pass: add frequency information
        for row in conc_table:
            row["l1_freq"] = l1_freq_counter.get(row["l1"], 0)
            row["r1_freq"] = r1_freq_counter.get(row["r1"], 0)

        return pl.DataFrame(conc_table)

    def frequency_analysis(
        self,
        time_column: str,
        group_by_columns: Optional[List[str]] = None,
        frequency: str = "monthly",
        sort_by_time: bool = True,
    ) -> pl.DataFrame:
        """
        Analyze frequency of records over time with optional grouping.

        Parameters
        ----------
        time_column : str
            Name of the column containing datetime/date values
        group_by_columns : List[str], optional
            Columns to group by (e.g., ['party', 'electorate']). If None, only time aggregation
        frequency : str, default "monthly"
            Time frequency for aggregation. Options: 'daily', 'weekly', 'monthly', 'yearly'
        sort_by_time : bool, default True
            Whether to sort results by time period

        Returns
        -------
        pl.DataFrame
            DataFrame with frequency analysis results

        Examples
        --------
        >>> # Monthly frequency by party
        >>> freq_df = df.text.frequency_analysis('created_at', ['party'], frequency='monthly')

        >>> # Daily frequency overall
        >>> daily_freq = df.text.frequency_analysis('created_at', frequency='daily')
        """
        if frequency not in ["daily", "weekly", "monthly", "yearly"]:
            raise ValueError(
                f"Unsupported frequency: {frequency}. Use 'daily', 'weekly', 'monthly', or 'yearly'"
            )

        # Create time period column based on frequency
        time_format = ""  # Initialize to avoid linting warnings
        if frequency == "daily":
            time_expr = pl.col(time_column).dt.date().alias("time_period")
            time_format = "%Y-%m-%d"
        elif frequency == "weekly":
            # Get Monday of the week
            time_expr = (
                pl.col(time_column).dt.truncate("1w").dt.date().alias("time_period")
            )
            time_format = "%Y-W%U"  # Year-Week format
        elif frequency == "monthly":
            time_expr = (
                pl.col(time_column).dt.truncate("1mo").dt.date().alias("time_period")
            )
            time_format = "%Y-%m"
        elif frequency == "yearly":
            time_expr = pl.col(time_column).dt.year().alias("time_period")
            time_format = "%Y"
        else:
            # This should never happen due to the check above, but keeps linter happy
            time_expr = pl.col(time_column).dt.date().alias("time_period")
            time_format = "%Y-%m-%d"

        # Build the aggregation
        df_with_period = self._df.with_columns(time_expr)

        # Determine grouping columns
        if group_by_columns is None:
            group_cols = ["time_period"]
        else:
            group_cols = ["time_period"] + group_by_columns

        # Perform aggregation
        result_df = df_with_period.group_by(group_cols).agg(
            [
                pl.len().alias("frequency_count"),
                pl.col(time_column).min().alias("period_start"),
                pl.col(time_column).max().alias("period_end"),
            ]
        )

        # Add formatted time period for display
        if frequency == "weekly":
            result_df = result_df.with_columns(
                [
                    pl.col("time_period")
                    .dt.strftime("%Y-W%W")
                    .alias("time_period_formatted")
                ]
            )
        elif frequency in ["daily", "monthly"]:
            result_df = result_df.with_columns(
                [
                    pl.col("time_period")
                    .dt.strftime(time_format)
                    .alias("time_period_formatted")
                ]
            )
        else:  # yearly
            result_df = result_df.with_columns(
                [pl.col("time_period").cast(pl.String).alias("time_period_formatted")]
            )

        # Sort by time if requested
        if sort_by_time:
            sort_cols = ["time_period"]
            if group_by_columns:
                sort_cols.extend(group_by_columns)
            result_df = result_df.sort(sort_cols)

        return result_df

    def to_docdataframe(self, document_column: Optional[str] = None):
        """
        Convert a regular polars DataFrame to a DocDataFrame.

        Parameters
        ----------
        document_column : str, optional
            Name of the column to use as the document column. If None, will try to auto-detect
            the string column with the longest average length.

        Returns
        -------
        DocDataFrame
            New DocDataFrame instance

        Examples
        --------
        >>> df = pl.DataFrame({'text': ['doc1', 'doc2'], 'id': [1, 2]})
        >>> doc_df = df.text.to_docdataframe(document_column='text')
        >>> doc_df = df.text.to_docdataframe()  # Auto-detect
        """

        return DocDataFrame(self._df, document_column=document_column)


@pl.api.register_lazyframe_namespace("text")
class TextLazyFrameNamespace:
    """Text processing namespace for polars LazyFrame"""

    def __init__(self, lf: pl.LazyFrame):
        self._lf = lf

    def tokenize(
        self, column: str, lowercase: bool = True, remove_punct: bool = True
    ) -> pl.LazyFrame:
        """Tokenize text column into list of tokens"""
        return self._lf.with_columns(
            pl.col(column)
            .text.tokenize(lowercase=lowercase, remove_punct=remove_punct)
            .alias(f"{column}_tokens")
        )

    def clean(
        self,
        column: str,
        lowercase: bool = True,
        remove_punct: bool = True,
        remove_digits: bool = False,
        remove_extra_whitespace: bool = True,
    ) -> pl.LazyFrame:
        """Clean text column with various options"""
        return self._lf.with_columns(
            pl.col(column)
            .text.clean(
                lowercase=lowercase,
                remove_punct=remove_punct,
                remove_digits=remove_digits,
                remove_extra_whitespace=remove_extra_whitespace,
            )
            .alias(f"{column}_clean")
        )

    def word_count(self, column: str) -> pl.LazyFrame:
        """Count words in text column"""
        return self._lf.with_columns(
            pl.col(column).text.word_count().alias(f"{column}_word_count")
        )

    def char_count(self, column: str) -> pl.LazyFrame:
        """Count characters in text column"""
        return self._lf.with_columns(
            pl.col(column).text.char_count().alias(f"{column}_char_count")
        )

    def sentence_count(self, column: str) -> pl.LazyFrame:
        """Count sentences in text column"""
        return self._lf.with_columns(
            pl.col(column).text.sentence_count().alias(f"{column}_sentence_count")
        )

    def ngrams(self, column: str, n: int = 2) -> pl.LazyFrame:
        """Extract n-grams from text column"""
        return self._lf.with_columns(
            pl.col(column).text.ngrams(n=n).alias(f"{column}_ngrams")
        )

    def contains_pattern(
        self, column: str, pattern: str, case_sensitive: bool = False
    ) -> pl.LazyFrame:
        """Check if text column contains a pattern"""
        return self._lf.with_columns(
            pl.col(column)
            .text.contains_pattern(pattern, case_sensitive=case_sensitive)
            .alias(f"{column}_contains")
        )

    def concordance(
        self,
        column: str,
        search_word: str,
        num_left_tokens: int = 10,
        num_right_tokens: int = 10,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> pl.DataFrame:
        """
        Generate a concordance table for a search word/pattern in a text column.
        Note: This method collects the LazyFrame to perform the concordance analysis.

        Parameters
        ----------
        column : str
            Name of the text column to search in
        search_word : str
            Word or pattern to search for
        num_left_tokens : int, default 10
            Number of tokens to include in left context
        num_right_tokens : int, default 10
            Number of tokens to include in right context
        regex : bool, default False
            Whether to treat search_word as a regex pattern
        case_sensitive : bool, default False
            Whether the search should be case sensitive

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: document_idx, left_context, matched_text, right_context, l1, l1_freq, r1, r1_freq
        """
        import re
        from collections import Counter

        from .text_utils import simple_tokenize

        if len(search_word) == 0:
            return pl.DataFrame(
                {
                    "document_idx": [],
                    "left_context": [],
                    "matched_text": [],
                    "right_context": [],
                    "l1": [],
                    "l1_freq": [],
                    "r1": [],
                    "r1_freq": [],
                },
                schema={
                    "document_idx": pl.Int32,
                    "left_context": pl.String,
                    "matched_text": pl.String,
                    "right_context": pl.String,
                    "l1": pl.String,
                    "l1_freq": pl.Int32,
                    "r1": pl.String,
                    "r1_freq": pl.Int32,
                },
            )

        # Create regex pattern
        pattern = search_word if regex else re.escape(search_word)
        flags = 0 if case_sensitive else re.IGNORECASE
        searcher = re.compile(pattern, flags)

        # Collect the LazyFrame and get the text column as a list
        collected_df = self._lf.collect()
        texts = collected_df[column].to_list()

        conc_table = []
        l1_tokens = []  # Collect all L1 tokens for frequency calculation
        r1_tokens = []  # Collect all R1 tokens for frequency calculation

        # First pass: collect all matches and extract L1/R1 tokens
        for idx, doc in enumerate(texts):
            if doc is None:
                continue

            for match in searcher.finditer(doc):
                matched_text = match.group(0)
                left_text = doc[: match.start()]
                right_text = doc[match.end() :]

                # Tokenize left and right contexts
                left_tokens = simple_tokenize(
                    left_text, lowercase=False, remove_punct=False
                )
                right_tokens = simple_tokenize(
                    right_text, lowercase=False, remove_punct=False
                )

                # Get the specified number of context tokens
                left_context_tokens = (
                    left_tokens[-num_left_tokens:] if num_left_tokens > 0 else []
                )
                right_context_tokens = (
                    right_tokens[:num_right_tokens] if num_right_tokens > 0 else []
                )

                # Extract L1 and R1 tokens (first left and first right)
                l1 = left_context_tokens[-1] if left_context_tokens else ""
                r1 = right_context_tokens[0] if right_context_tokens else ""

                conc_table.append(
                    {
                        "document_idx": idx,
                        "left_context": " ".join(left_context_tokens),
                        "matched_text": matched_text,
                        "right_context": " ".join(right_context_tokens),
                        "l1": l1,
                        "r1": r1,
                    }
                )

                # Collect for frequency calculation
                if l1:
                    l1_tokens.append(l1)
                if r1:
                    r1_tokens.append(r1)

        if len(conc_table) == 0:
            return pl.DataFrame(
                {
                    "document_idx": [],
                    "left_context": [],
                    "matched_text": [],
                    "right_context": [],
                    "l1": [],
                    "l1_freq": [],
                    "r1": [],
                    "r1_freq": [],
                },
                schema={
                    "document_idx": pl.Int32,
                    "left_context": pl.String,
                    "matched_text": pl.String,
                    "right_context": pl.String,
                    "l1": pl.String,
                    "l1_freq": pl.Int32,
                    "r1": pl.String,
                    "r1_freq": pl.Int32,
                },
            )

        # Calculate frequencies
        l1_freq_counter = Counter(l1_tokens)
        r1_freq_counter = Counter(r1_tokens)

        # Second pass: add frequency information
        for row in conc_table:
            row["l1_freq"] = l1_freq_counter.get(row["l1"], 0)
            row["r1_freq"] = r1_freq_counter.get(row["r1"], 0)

        return pl.DataFrame(conc_table)

    def frequency_analysis(
        self,
        time_column: str,
        group_by_columns: Optional[List[str]] = None,
        frequency: str = "monthly",
        sort_by_time: bool = True,
    ) -> pl.DataFrame:
        """
        Analyze frequency of records over time with optional grouping.

        Parameters
        ----------
        time_column : str
            Name of the column containing datetime/date values
        group_by_columns : List[str], optional
            Columns to group by (e.g., ['party', 'electorate']). If None, only time aggregation
        frequency : str, default "monthly"
            Time frequency for aggregation. Options: 'daily', 'weekly', 'monthly', 'yearly'
        sort_by_time : bool, default True
            Whether to sort results by time period

        Returns
        -------
        pl.DataFrame
            DataFrame with frequency analysis results

        Examples
        --------
        >>> # Monthly frequency by party
        >>> freq_df = lf.text.frequency_analysis('created_at', ['party'], frequency='monthly')

        >>> # Daily frequency overall
        >>> daily_freq = lf.text.frequency_analysis('created_at', frequency='daily')
        """
        if frequency not in ["daily", "weekly", "monthly", "yearly"]:
            raise ValueError(
                f"Unsupported frequency: {frequency}. Use 'daily', 'weekly', 'monthly', or 'yearly'"
            )

        # Create time period column based on frequency
        time_format = ""  # Initialize to avoid linting warnings
        if frequency == "daily":
            time_expr = pl.col(time_column).dt.date().alias("time_period")
            time_format = "%Y-%m-%d"
        elif frequency == "weekly":
            # Get Monday of the week
            time_expr = (
                pl.col(time_column).dt.truncate("1w").dt.date().alias("time_period")
            )
            time_format = "%Y-W%U"  # Year-Week format
        elif frequency == "monthly":
            time_expr = (
                pl.col(time_column).dt.truncate("1mo").dt.date().alias("time_period")
            )
            time_format = "%Y-%m"
        elif frequency == "yearly":
            time_expr = pl.col(time_column).dt.year().alias("time_period")
            time_format = "%Y"
        else:
            # This should never happen due to the check above, but keeps linter happy
            time_expr = pl.col(time_column).dt.date().alias("time_period")
            time_format = "%Y-%m-%d"

        # Build the aggregation
        lf_with_period = self._lf.with_columns(time_expr)

        # Determine grouping columns
        if group_by_columns is None:
            group_cols = ["time_period"]
        else:
            group_cols = ["time_period"] + group_by_columns

        # Perform aggregation
        result_lf = lf_with_period.group_by(group_cols).agg(
            [
                pl.len().alias("frequency_count"),
                pl.col(time_column).min().alias("period_start"),
                pl.col(time_column).max().alias("period_end"),
            ]
        )

        # Add formatted time period for display
        if frequency == "weekly":
            result_lf = result_lf.with_columns(
                [
                    pl.col("time_period")
                    .dt.strftime("%Y-W%W")
                    .alias("time_period_formatted")
                ]
            )
        elif frequency in ["daily", "monthly"]:
            result_lf = result_lf.with_columns(
                [
                    pl.col("time_period")
                    .dt.strftime(time_format)
                    .alias("time_period_formatted")
                ]
            )
        else:  # yearly
            result_lf = result_lf.with_columns(
                [pl.col("time_period").cast(pl.String).alias("time_period_formatted")]
            )

        # Sort by time if requested
        if sort_by_time:
            sort_cols = ["time_period"]
            if group_by_columns:
                sort_cols.extend(group_by_columns)
            result_lf = result_lf.sort(sort_cols)

        # Collect to DataFrame and return
        return result_lf.collect()

    def to_doclazyframe(self, document_column: Optional[str] = None):
        """
        Convert a regular polars LazyFrame to a DocLazyFrame.

        Parameters
        ----------
        document_column : str, optional
            Name of the column to use as the document column. If None, will try to auto-detect
            the string column with the longest average length.

        Returns
        -------
        DocLazyFrame
            New DocLazyFrame instance

        Examples
        --------
        >>> lf = pl.LazyFrame({'text': ['doc1', 'doc2'], 'id': [1, 2]})
        >>> doc_lf = lf.text.to_doclazyframe(document_column='text')
        >>> doc_lf = lf.text.to_doclazyframe()  # Auto-detect
        """
        return DocLazyFrame(self._lf, document_column=document_column)

    def to_docdataframe(self, document_column: Optional[str] = None):
        """
        Convert a regular polars LazyFrame to a DocDataFrame by collecting first.

        Parameters
        ----------
        document_column : str, optional
            Name of the column to use as the document column. If None, will try to auto-detect
            the string column with the longest average length.

        Returns
        -------
        DocDataFrame
            New DocDataFrame instance (collected from LazyFrame)

        Examples
        --------
        >>> lf = pl.LazyFrame({'text': ['doc1', 'doc2'], 'id': [1, 2]})
        >>> doc_df = lf.text.to_docdataframe(document_column='text')
        >>> doc_df = lf.text.to_docdataframe()  # Auto-detect
        """
        return DocDataFrame(self._lf.collect(), document_column=document_column)
