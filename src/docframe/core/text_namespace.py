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
    concordance_elements,
    quotation_elements,
    remove_stopwords,
    sentence_count,
    tokenize,
    word_count,
)


@pl.api.register_expr_namespace("text")
class TextExprNamespace:
    """Text processing namespace for polars expressions"""

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def tokenize(
        self, lowercase: bool = True, remove_punct: bool = True, explode: bool = False
    ) -> pl.Expr:
        """Tokenize text into list of tokens"""

        _tokenize = partial(tokenize, lowercase=lowercase, remove_punct=remove_punct)

        results = self._expr.map_elements(_tokenize, return_dtype=pl.List(pl.String))
        if explode:
            return results.list.explode()
        else:
            return results

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

    def concordance(
        self,
        search_word: str,
        num_left_tokens: int = 10,
        num_right_tokens: int = 10,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> pl.Expr:
        """Element-wise concordance returning list-of-dicts per row.

        Returns List[Struct] per element. Users can call .list.explode() if needed.
        """

        def _conc(text: Optional[str]):
            return concordance_elements(
                text,
                search_word,
                num_left_tokens=num_left_tokens,
                num_right_tokens=num_right_tokens,
                regex=regex,
                case_sensitive=case_sensitive,
            )

        # Map to list of Python dicts; Polars will infer to List(Struct(...))
        expr = self._expr.map_elements(
            _conc,
            return_dtype=pl.List(
                pl.Struct([
                    pl.Field("left_context", pl.String),
                    pl.Field("matched_text", pl.String),
                    pl.Field("right_context", pl.String),
                    pl.Field("start_idx", pl.Int64),
                    pl.Field("end_idx", pl.Int64),
                    pl.Field("l1", pl.String),
                    pl.Field("r1", pl.String),
                ])
            ),
        )

        return expr

    def quotation(self) -> pl.Expr:
        """Element-wise quotation extraction returning list-of-dicts per row.

        Each element is a list of structs with fields:
        speaker, speaker_start_idx, speaker_end_idx,
        quote, quote_start_idx, quote_end_idx,
        verb, verb_start_idx, verb_end_idx, quote_type
        """

        def _quotes(text: Optional[str]):
            return quotation_elements(text)

        return self._expr.map_elements(
            _quotes,
            return_dtype=pl.List(
                pl.Struct([
                    pl.Field("speaker", pl.String),
                    pl.Field("speaker_start_idx", pl.Int64),
                    pl.Field("speaker_end_idx", pl.Int64),
                    pl.Field("quote", pl.String),
                    pl.Field("quote_start_idx", pl.Int64),
                    pl.Field("quote_end_idx", pl.Int64),
                    pl.Field("verb", pl.String),
                    pl.Field("verb_start_idx", pl.Int64),
                    pl.Field("verb_end_idx", pl.Int64),
                    pl.Field("quote_type", pl.String),
                    pl.Field("quote_token_count", pl.Int64),
                    pl.Field("is_floating_quote", pl.Boolean),
                ])
            ),
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

    #     tokenize, lowercase=lowercase, remove_punct=remove_punct
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

    def concordance(
        self,
        search_word: str,
        num_left_tokens: int = 10,
        num_right_tokens: int = 10,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> pl.Series:
        """Series-level concordance returning List[Struct] per element.

        Use .list.explode() on the result to get one row per match if desired.
        """
        return (
            self._series.to_frame()
            .select(
                pl.col(self._series.name)
                .text.concordance(
                    search_word,
                    num_left_tokens=num_left_tokens,
                    num_right_tokens=num_right_tokens,
                    regex=regex,
                    case_sensitive=case_sensitive,
                )
                .alias(self._series.name)
            )
            .to_series()
        )

    def quotation(self) -> pl.Series:
        """Series-level quotation extraction returning List[Struct] per element."""
        return (
            self._series.to_frame()
            .select(pl.col(self._series.name).text.quotation().alias(self._series.name))
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

    def quotation(
        self, column: str, *, explode: bool = False, unnest: bool = False
    ) -> pl.DataFrame:
        """Extract quotations from a text column using heuristics.

        Behavior:
        - explode=False: produce a single '__quotation__' column (List[Struct]).
          Requires unnest=True; otherwise raises ValueError.
        - explode=True: produce an expanded table with list.explode, keeping original
          columns; if unnest=True, expand the struct fields into separate columns,
          otherwise keep the '__quotation__' struct column as-is.
        """
        tmp = self._df.with_columns(
            pl.col(column).text.quotation().alias("__quotation__")
        )

        if not explode:
            if not unnest:
                raise ValueError("explode=False requires unnest=True for quotation")
            # Return only the special column
            return tmp.select([pl.col("__quotation__")])

        # explode=True: keep original columns
        exploded = tmp.explode("__quotation__")
        if unnest:
            return exploded.unnest("__quotation__")
        return exploded

    def concordance(
        self,
        column: str,
        search_word: str,
        num_left_tokens: int = 10,
        num_right_tokens: int = 10,
        regex: bool = False,
        case_sensitive: bool = False,
        *,
        explode: bool = False,
        unnest: bool = False,
    ) -> pl.DataFrame:
        if unnest and not explode:
            raise ValueError("unnest=True requires explode=True for concordance")

        # Special-case: empty search word
        if len(search_word) == 0:
            if not explode:
                # Return a single column with empty lists per row
                return self._df.select([
                    pl.lit(
                        pl.Series(
                            [],
                            dtype=pl.Struct([
                                pl.Field("left_context", pl.String),
                                pl.Field("matched_text", pl.String),
                                pl.Field("right_context", pl.String),
                                pl.Field("start_idx", pl.Int64),
                                pl.Field("end_idx", pl.Int64),
                                pl.Field("l1", pl.String),
                                pl.Field("r1", pl.String),
                            ]),
                        )
                    )
                    .repeat_by(pl.len())
                    .explode()
                    .alias("__concordance__")
                ])
            # explode=True
            if unnest:
                # Empty expanded table with expected columns
                return pl.DataFrame({
                    "left_context": pl.Series([], dtype=pl.String),
                    "matched_text": pl.Series([], dtype=pl.String),
                    "right_context": pl.Series([], dtype=pl.String),
                    "start_idx": pl.Series([], dtype=pl.Int64),
                    "end_idx": pl.Series([], dtype=pl.Int64),
                    "l1": pl.Series([], dtype=pl.String),
                    "l1_freq": pl.Series([], dtype=pl.Int32),
                    "r1": pl.Series([], dtype=pl.String),
                    "r1_freq": pl.Series([], dtype=pl.Int32),
                })
            # explode=True, unnest=False -> keep original columns with zero rows
            # Achieve by filtering to zero rows cheaply
            return self._df.filter(pl.lit(False)).with_columns([
                pl.col(column)
                .map_elements(
                    lambda _: None,
                    return_dtype=pl.Struct([
                        pl.Field("left_context", pl.String),
                        pl.Field("matched_text", pl.String),
                        pl.Field("right_context", pl.String),
                        pl.Field("start_idx", pl.Int64),
                        pl.Field("end_idx", pl.Int64),
                        pl.Field("l1", pl.String),
                        pl.Field("r1", pl.String),
                    ]),
                )
                .alias("__concordance__")
            ])

        tmp = self._df.with_columns(
            pl.col(column)
            .text.concordance(
                search_word,
                num_left_tokens=num_left_tokens,
                num_right_tokens=num_right_tokens,
                regex=regex,
                case_sensitive=case_sensitive,
            )
            .alias("__concordance__")
        )

        # Handle explode=False case
        if not explode:
            return tmp.select([pl.col("__concordance__")])

        # explode == True: keep original columns
        exploded = tmp.explode("__concordance__")
        if not unnest:
            return exploded

        # unnest the struct and compute l1/r1 frequencies
        df = exploded.unnest("__concordance__")
        if df.height == 0:
            # Ensure frequency columns exist even for empty results
            return df.with_columns([
                pl.lit(None).cast(pl.Int32).alias("l1_freq"),
                pl.lit(None).cast(pl.Int32).alias("r1_freq"),
            ])

        l1_counts = df.group_by("l1").agg(pl.len().alias("l1_freq"))
        r1_counts = df.group_by("r1").agg(pl.len().alias("r1_freq"))

        df = df.join(l1_counts, on="l1", how="left").join(
            r1_counts, on="r1", how="left"
        )
        df = df.with_columns([
            pl.col("l1_freq").fill_null(0).cast(pl.Int32),
            pl.col("r1_freq").fill_null(0).cast(pl.Int32),
        ])
        return df

    def sequential_analysis(
        self,
        time_column: str,
        group_by_columns: Optional[List[str]] = None,
        frequency: str = "monthly",
        sort_by_time: bool = True,
        column_type: str = "datetime",
        numeric_origin: Optional[float] = None,
        numeric_interval: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Analyze sequential records over time with optional grouping.

        Parameters
        ----------
        time_column : str
            Name of the column containing datetime/date values or numeric values
        group_by_columns : List[str], optional
            Columns to group by (e.g., ['party', 'electorate']). If None, only time aggregation
        frequency : str, default "monthly"
            Time frequency for aggregation. Options: 'hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
        sort_by_time : bool, default True
            Whether to sort results by time period
        column_type : str, default "datetime"
            Column interpretation mode. Options: 'datetime' or 'numeric'
        numeric_origin : float, optional
            Starting value for numeric bins (defaults to column minimum)
        numeric_interval : float, optional
            Bin width for numeric columns (required when column_type="numeric")

        Returns
        -------
        pl.DataFrame
            DataFrame with sequential analysis results

        Examples
        --------
        >>> # Monthly sequences by party
        >>> seq_df = df.text.sequential_analysis('created_at', ['party'], frequency='monthly')

        >>> # Daily sequences overall
        >>> daily_seq = df.text.sequential_analysis('created_at', frequency='daily')
        """

        normalized_column_type = (column_type or "datetime").lower()
        if normalized_column_type not in {"datetime", "numeric"}:
            raise ValueError(
                "Unsupported column_type. Use 'datetime' or 'numeric' for sequential analysis"
            )

        valid_frequencies = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
        if normalized_column_type == "datetime" and frequency not in valid_frequencies:
            raise ValueError(
                "Unsupported frequency: {}. Use 'hourly', 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'".format(
                    frequency
                )
            )

        df_with_period = self._df
        time_format = ""
        numeric_interval_value: Optional[float] = None
        numeric_origin_value: Optional[float] = None

        if normalized_column_type == "datetime":
            if frequency == "hourly":
                time_expr = pl.col(time_column).dt.truncate("1h").alias("time_period")
                time_format = "%Y-%m-%d %H:%M"
            elif frequency == "daily":
                time_expr = pl.col(time_column).dt.date().alias("time_period")
                time_format = "%Y-%m-%d"
            elif frequency == "weekly":
                time_expr = (
                    pl.col(time_column).dt.truncate("1w").dt.date().alias("time_period")
                )
                time_format = "%Y-W%U"
            elif frequency == "monthly":
                time_expr = (
                    pl.col(time_column).dt.truncate("1mo").dt.date().alias("time_period")
                )
                time_format = "%Y-%m"
            elif frequency == "quarterly":
                time_expr = (
                    pl.col(time_column).dt.truncate("3mo").dt.date().alias("time_period")
                )
                time_format = "%Y-Q"
            elif frequency == "yearly":
                time_expr = pl.col(time_column).dt.truncate("1y").dt.date().alias("time_period")
                time_format = "%Y"
            else:  # pragma: no cover
                time_expr = pl.col(time_column).dt.date().alias("time_period")
                time_format = "%Y-%m-%d"

            df_with_period = df_with_period.with_columns(time_expr)
        else:
            if numeric_interval is None or numeric_interval <= 0:
                raise ValueError("numeric_interval must be a positive number for numeric sequential analysis")
            numeric_interval_value = float(numeric_interval)
            if numeric_origin is not None:
                numeric_origin_value = float(numeric_origin)
            else:
                origin_series = (
                    df_with_period.select(pl.col(time_column).cast(pl.Float64()).min()).to_series()
                )
                numeric_origin_value = origin_series[0] if len(origin_series) else None
            if numeric_origin_value is None:
                raise ValueError("Unable to determine numeric_origin from the provided data")

            df_with_period = df_with_period.with_columns([
                pl.col(time_column).cast(pl.Float64()).alias("__numeric_value__"),
            ])
            df_with_period = df_with_period.with_columns([
                (
                    (pl.col("__numeric_value__") - pl.lit(numeric_origin_value))
                    / pl.lit(numeric_interval_value)
                )
                .floor()
                .cast(pl.Int64)
                .alias("__numeric_bin__"),
            ])
            df_with_period = df_with_period.with_columns([
                (
                    pl.lit(numeric_origin_value)
                    + pl.col("__numeric_bin__").cast(pl.Float64) * pl.lit(numeric_interval_value)
                ).alias("time_period"),
            ])

        # Determine grouping columns
        if group_by_columns is None:
            group_cols = ["time_period"]
        else:
            group_cols = ["time_period"] + group_by_columns

        # Perform aggregation
        result_df = df_with_period.group_by(group_cols).agg([
            pl.len().alias("sequential_count"),
            pl.col(time_column).min().alias("period_start"),
            pl.col(time_column).max().alias("period_end"),
        ])

        # Add formatted time period for display
        if normalized_column_type == "datetime":
            if frequency == "weekly":
                result_df = result_df.with_columns([
                    pl.col("time_period")
                    .dt.strftime("%Y-W%W")
                    .alias("time_period_formatted")
                ])
            elif frequency == "quarterly":
                result_df = result_df.with_columns([
                    pl.col("time_period").dt.year().alias("__year__"),
                    (
                        (pl.col("time_period").dt.month() - 1)
                        .floordiv(3)
                        .add(1)
                    ).alias("__quarter__"),
                ])
                result_df = result_df.with_columns([
                    pl.format(
                        "{}-Q{}",
                        pl.col("__year__"),
                        pl.col("__quarter__"),
                    ).alias("time_period_formatted")
                ]).drop(["__year__", "__quarter__"])
            elif frequency == "yearly":
                result_df = result_df.with_columns([
                    pl.col("time_period").dt.strftime(time_format).alias("time_period_formatted")
                ])
            else:
                result_df = result_df.with_columns([
                    pl.col("time_period")
                    .dt.strftime(time_format)
                    .alias("time_period_formatted")
                ])
        else:
            interval_lit = pl.lit(numeric_interval_value)
            result_df = result_df.with_columns([
                pl.col("time_period").round(6).alias("time_period"),
                (
                    pl.col("time_period") + interval_lit
                ).alias("__numeric_period_end__"),
            ])

            def _format_numeric(value: Optional[float]) -> Optional[str]:
                if value is None:
                    return None
                return format(value, ".6g")

            result_df = result_df.with_columns([
                pl.col("time_period")
                .map_elements(_format_numeric, return_dtype=pl.String)
                .alias("__numeric_period_label_start__"),
                pl.col("__numeric_period_end__")
                .map_elements(_format_numeric, return_dtype=pl.String)
                .alias("__numeric_period_label_end__"),
            ])
            result_df = result_df.with_columns([
                pl.format(
                    "[{}, {})",
                    pl.col("__numeric_period_label_start__"),
                    pl.col("__numeric_period_label_end__"),
                ).alias("time_period_formatted")
            ]).drop([
                "__numeric_period_end__",
                "__numeric_period_label_start__",
                "__numeric_period_label_end__",
            ])

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

    def quotation(
        self, column: str, *, explode: bool = False, unnest: bool = False
    ) -> pl.DataFrame:
        """Extract quotations from a text column on a LazyFrame.

        Collects and delegates to DataFrame namespace with explode/unnest.
        """
        collected = self._lf.collect()
        return collected.text.quotation(column, explode=explode, unnest=unnest)

    def concordance(
        self,
        column: str,
        search_word: str,
        num_left_tokens: int = 10,
        num_right_tokens: int = 10,
        regex: bool = False,
        case_sensitive: bool = False,
        *,
        explode: bool = False,
        unnest: bool = False,
    ) -> pl.DataFrame:
        collected = self._lf.collect()
        return collected.text.concordance(
            column,
            search_word,
            num_left_tokens=num_left_tokens,
            num_right_tokens=num_right_tokens,
            regex=regex,
            case_sensitive=case_sensitive,
            explode=explode,
            unnest=unnest,
        )

    def sequential_analysis(
        self,
        time_column: str,
        group_by_columns: Optional[List[str]] = None,
        frequency: str = "monthly",
        sort_by_time: bool = True,
        column_type: str = "datetime",
        numeric_origin: Optional[float] = None,
        numeric_interval: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Analyze sequential records over time with optional grouping.

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
            DataFrame with sequential analysis results

        Examples
        --------
        >>> # Monthly sequences by party
        >>> seq_df = lf.text.sequential_analysis('created_at', ['party'], frequency='monthly')

        >>> # Daily sequences overall
        >>> daily_seq = lf.text.sequential_analysis('created_at', frequency='daily')
        """
        normalized_column_type = (column_type or "datetime").lower()
        if normalized_column_type not in {"datetime", "numeric"}:
            raise ValueError(
                "Unsupported column_type. Use 'datetime' or 'numeric' for sequential analysis"
            )

        valid_frequencies = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
        if normalized_column_type == "datetime" and frequency not in valid_frequencies:
            raise ValueError(
                "Unsupported frequency: {}. Use 'hourly', 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'".format(
                    frequency
                )
            )

        lf_with_period = self._lf
        time_format = ""
        numeric_interval_value: Optional[float] = None
        numeric_origin_value: Optional[float] = None

        if normalized_column_type == "datetime":
            if frequency == "hourly":
                time_expr = pl.col(time_column).dt.truncate("1h").alias("time_period")
                time_format = "%Y-%m-%d %H:%M"
            elif frequency == "daily":
                time_expr = pl.col(time_column).dt.date().alias("time_period")
                time_format = "%Y-%m-%d"
            elif frequency == "weekly":
                time_expr = (
                    pl.col(time_column).dt.truncate("1w").dt.date().alias("time_period")
                )
                time_format = "%Y-W%U"
            elif frequency == "monthly":
                time_expr = (
                    pl.col(time_column).dt.truncate("1mo").dt.date().alias("time_period")
                )
                time_format = "%Y-%m"
            elif frequency == "quarterly":
                time_expr = (
                    pl.col(time_column).dt.truncate("3mo").dt.date().alias("time_period")
                )
                time_format = "%Y-Q"
            elif frequency == "yearly":
                time_expr = pl.col(time_column).dt.truncate("1y").dt.date().alias("time_period")
                time_format = "%Y"
            else:  # pragma: no cover
                time_expr = pl.col(time_column).dt.date().alias("time_period")
                time_format = "%Y-%m-%d"

            lf_with_period = lf_with_period.with_columns(time_expr)
        else:
            if numeric_interval is None or numeric_interval <= 0:
                raise ValueError("numeric_interval must be a positive number for numeric sequential analysis")
            numeric_interval_value = float(numeric_interval)
            if numeric_origin is not None:
                numeric_origin_value = float(numeric_origin)
            else:
                origin_series = (
                    self._lf.select(pl.col(time_column).cast(pl.Float64()).min())
                    .collect()
                    .to_series()
                )
                numeric_origin_value = origin_series[0] if len(origin_series) else None
            if numeric_origin_value is None:
                raise ValueError("Unable to determine numeric_origin from the provided data")

            lf_with_period = lf_with_period.with_columns([
                pl.col(time_column).cast(pl.Float64()).alias("__numeric_value__"),
            ])
            lf_with_period = lf_with_period.with_columns([
                (
                    (pl.col("__numeric_value__") - pl.lit(numeric_origin_value))
                    / pl.lit(numeric_interval_value)
                )
                .floor()
                .cast(pl.Int64)
                .alias("__numeric_bin__"),
            ])
            lf_with_period = lf_with_period.with_columns([
                (
                    pl.lit(numeric_origin_value)
                    + pl.col("__numeric_bin__").cast(pl.Float64) * pl.lit(numeric_interval_value)
                ).alias("time_period"),
            ])

        # Determine grouping columns
        if group_by_columns is None:
            group_cols = ["time_period"]
        else:
            group_cols = ["time_period"] + group_by_columns

        # Perform aggregation
        result_lf = lf_with_period.group_by(group_cols).agg([
            pl.len().alias("sequential_count"),
            pl.col(time_column).min().alias("period_start"),
            pl.col(time_column).max().alias("period_end"),
        ])

        # Add formatted time period for display
        if normalized_column_type == "datetime":
            if frequency == "weekly":
                result_lf = result_lf.with_columns([
                    pl.col("time_period")
                    .dt.strftime("%Y-W%W")
                    .alias("time_period_formatted")
                ])
            elif frequency == "quarterly":
                result_lf = result_lf.with_columns([
                    pl.col("time_period").dt.year().alias("__year__"),
                    (
                        (pl.col("time_period").dt.month() - 1)
                        .floordiv(3)
                        .add(1)
                    ).alias("__quarter__"),
                ])
                result_lf = result_lf.with_columns([
                    pl.format(
                        "{}-Q{}",
                        pl.col("__year__"),
                        pl.col("__quarter__"),
                    ).alias("time_period_formatted")
                ]).drop(["__year__", "__quarter__"])
            elif frequency == "yearly":
                result_lf = result_lf.with_columns([
                    pl.col("time_period").dt.strftime(time_format).alias("time_period_formatted")
                ])
            else:
                result_lf = result_lf.with_columns([
                    pl.col("time_period")
                    .dt.strftime(time_format)
                    .alias("time_period_formatted")
                ])
        else:
            interval_lit = pl.lit(numeric_interval_value)
            result_lf = result_lf.with_columns([
                pl.col("time_period").round(6).alias("time_period"),
                (
                    pl.col("time_period") + interval_lit
                ).alias("__numeric_period_end__"),
            ])

            def _format_numeric(value: Optional[float]) -> Optional[str]:
                if value is None:
                    return None
                return format(value, ".6g")

            result_lf = result_lf.with_columns([
                pl.col("time_period")
                .map_elements(_format_numeric, return_dtype=pl.String)
                .alias("__numeric_period_label_start__"),
                pl.col("__numeric_period_end__")
                .map_elements(_format_numeric, return_dtype=pl.String)
                .alias("__numeric_period_label_end__"),
            ])
            result_lf = result_lf.with_columns([
                pl.format(
                    "[{}, {})",
                    pl.col("__numeric_period_label_start__"),
                    pl.col("__numeric_period_label_end__"),
                ).alias("time_period_formatted")
            ]).drop([
                "__numeric_period_end__",
                "__numeric_period_label_start__",
                "__numeric_period_label_end__",
            ])

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
