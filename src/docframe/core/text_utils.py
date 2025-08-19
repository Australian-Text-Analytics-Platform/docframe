"""
Text processing utilities
"""

import re
import string
from functools import reduce
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

try:  # Lazy optional BERTopic dependency
    import importlib

    _bertopic_mod = importlib.import_module("bertopic")  # noqa: F401
    _HAS_BERTOPIC = True
except Exception:  # pragma: no cover
    _HAS_BERTOPIC = False
from sklearn.preprocessing import MinMaxScaler

try:  # Optional dependency: if unavailable, we fall back to PCA
    from umap import UMAP  # type: ignore

    _HAS_UMAP = True
except Exception:  # pragma: no cover - environment without umap
    UMAP = None  # type: ignore
    _HAS_UMAP = False


def simple_tokenize(
    text: str, lowercase: bool = True, remove_punct: bool = True
) -> List[str]:
    """Simple tokenization using regex"""
    if not isinstance(text, str):
        return []

    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()

    # Remove punctuation if requested
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))

    # Split on whitespace
    tokens = text.split()
    return [token.strip() for token in tokens if token.strip()]


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_digits: bool = False,
    remove_extra_whitespace: bool = True,
) -> str:
    """Clean text with various options"""
    if not isinstance(text, str):
        return ""

    result = text

    if lowercase:
        result = result.lower()

    if remove_punct:
        result = result.translate(str.maketrans("", "", string.punctuation))

    if remove_digits:
        result = re.sub(r"\d+", "", result)

    if remove_extra_whitespace:
        result = re.sub(r"\s+", " ", result).strip()

    return result


def word_count(text: str) -> int:
    """Count words in text"""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def char_count(text: str) -> int:
    """Count characters in text"""
    if not isinstance(text, str):
        return 0
    return len(text)


def sentence_count(text: str) -> int:
    """Count sentences in text (simple approach)"""
    if not isinstance(text, str):
        return 0
    # Simple sentence splitting on common sentence endings
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """Extract n-grams from text"""
    if not isinstance(text, str):
        return []

    tokens = simple_tokenize(text)
    if len(tokens) < n:
        return []

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i : i + n])
        ngrams.append(ngram)

    return ngrams


def contains_pattern(text: str, pattern: str, case_sensitive: bool = False) -> bool:
    """Check if text contains a pattern"""
    if not isinstance(text, str) or not isinstance(pattern, str):
        return False

    flags = 0 if case_sensitive else re.IGNORECASE
    return bool(re.search(pattern, text, flags))


def remove_stopwords(
    tokens: List[str], stopwords: Optional[List[str]] = None
) -> List[str]:
    """Remove stopwords from token list"""
    if stopwords is None:
        # Basic English stopwords
        stopwords_set = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
            "must",
            "this",
            "that",
            "these",
            "those",
        }
    else:
        stopwords_set = set(stopwords)

    return [token for token in tokens if token.lower() not in stopwords_set]


def _calculate_log_likelihood_and_effect_size(
    freq_tables: List[Dict[str, int]],
) -> pl.DataFrame:
    """
    Calculate log likelihood and effect size statistics for frequency tables using Polars.

    Based on the implementation from:
    - https://ucrel.lancs.ac.uk/llwizard.html
    - Rayson, P. and Garside, R. (2000)

    Parameters
    ----------
    freq_tables : List[Dict[str, int]]
        List of frequency dictionaries (usually 2 for comparison)

    Returns
    -------
    pl.DataFrame
        DataFrame with statistical measures
    """
    if len(freq_tables) != 2:
        raise ValueError(
            "Log likelihood calculation requires exactly 2 frequency tables for comparison"
        )

    # Get all tokens and create DataFrame from frequency dictionaries
    all_tokens = sorted(set().union(*freq_tables))

    # Create data for DataFrame
    data = []
    for token in all_tokens:
        freq1 = freq_tables[0].get(token, 0)
        freq2 = freq_tables[1].get(token, 0)
        data.append({"token": token, "freq_corpus_0": freq1, "freq_corpus_1": freq2})

    # Create Polars DataFrame
    df = pl.DataFrame(data)

    # Calculate corpus-level statistics
    df = df.with_columns([
        (pl.col("freq_corpus_0") + pl.col("freq_corpus_1")).alias("total_freq"),
        pl.col("freq_corpus_0").sum().alias("corpus_0_total"),
        pl.col("freq_corpus_1").sum().alias("corpus_1_total"),
    ])

    # Calculate grand total
    grand_total = df.select(
        pl.col("corpus_0_total").first() + pl.col("corpus_1_total").first()
    ).item()

    # Calculate expected frequencies
    df = df.with_columns([
        (pl.col("total_freq") * pl.col("corpus_0_total") / grand_total).alias(
            "expected_0"
        ),
        (pl.col("total_freq") * pl.col("corpus_1_total") / grand_total).alias(
            "expected_1"
        ),
    ])

    # Calculate log likelihood ratios with safe division (avoid log(0))
    df = df.with_columns([
        # Use observed * log(observed/expected) formula for log likelihood
        pl.when(pl.col("freq_corpus_0") > 0)
        .then(
            pl.col("freq_corpus_0")
            * (
                pl.col("freq_corpus_0") / pl.max_horizontal("expected_0", pl.lit(1e-10))
            ).log()
        )
        .otherwise(0.0)
        .alias("ll_0"),
        pl.when(pl.col("freq_corpus_1") > 0)
        .then(
            pl.col("freq_corpus_1")
            * (
                pl.col("freq_corpus_1") / pl.max_horizontal("expected_1", pl.lit(1e-10))
            ).log()
        )
        .otherwise(0.0)
        .alias("ll_1"),
    ])

    # Calculate G2 log likelihood statistic
    df = df.with_columns([
        (2 * (pl.col("ll_0") + pl.col("ll_1"))).alias("log_likelihood_llv")
    ])

    # Calculate Bayes Factor (BIC)
    dof = 1  # degrees of freedom for 2x2 contingency table
    df = df.with_columns([
        (pl.col("log_likelihood_llv") - (dof * pl.lit(grand_total).log())).alias(
            "bayes_factor_bic"
        )
    ])

    # Calculate Effect Size for Log Likelihood (ELL)
    df = df.with_columns([
        pl.min_horizontal("expected_0", "expected_1").alias("min_expected")
    ])

    df = df.with_columns([
        pl.when(pl.col("min_expected") > 0)
        .then(
            pl.col("log_likelihood_llv")
            / (grand_total * pl.max_horizontal("min_expected", pl.lit(1e-10)).log())
        )
        .otherwise(0.0)
        .alias("effect_size_ell")
    ])

    # Add significance indicators based on critical values
    df = df.with_columns([
        pl.when(pl.col("log_likelihood_llv") >= 15.13)
        .then(pl.lit("****"))  # p < 0.0001
        .when(pl.col("log_likelihood_llv") >= 10.83)
        .then(pl.lit("***"))  # p < 0.001
        .when(pl.col("log_likelihood_llv") >= 6.63)
        .then(pl.lit("**"))  # p < 0.01
        .when(pl.col("log_likelihood_llv") >= 3.84)
        .then(pl.lit("*"))  # p < 0.05
        .otherwise(pl.lit(""))  # not significant
        .alias("significance")
    ])

    # Return only the key statistical measures, with token as index
    result = df.select([
        "token",
        "freq_corpus_0",  # O1 - observed frequency in corpus 1
        "freq_corpus_1",  # O2 - observed frequency in corpus 2
        "expected_0",  # Expected frequency in corpus 1
        "expected_1",  # Expected frequency in corpus 2
        "corpus_0_total",  # Total tokens in corpus 1
        "corpus_1_total",  # Total tokens in corpus 2
        "log_likelihood_llv",
        "bayes_factor_bic",
        "effect_size_ell",
        "significance",
    ])

    # Add percentage columns and additional statistics
    result = result.with_columns([
        # %1 and %2 - percentage of token in each corpus
        (pl.col("freq_corpus_0") / pl.col("corpus_0_total") * 100).alias(
            "percent_corpus_0"
        ),
        (pl.col("freq_corpus_1") / pl.col("corpus_1_total") * 100).alias(
            "percent_corpus_1"
        ),
        # %DIFF - percentage difference between corpora
        (
            (pl.col("freq_corpus_0") / pl.col("corpus_0_total"))
            - (pl.col("freq_corpus_1") / pl.col("corpus_1_total"))
        ).alias("percent_diff"),
        # Relative Risk (RRisk) - ratio of proportions
        pl.when(pl.col("freq_corpus_1") > 0)
        .then(
            (pl.col("freq_corpus_0") / pl.col("corpus_0_total"))
            / (pl.col("freq_corpus_1") / pl.col("corpus_1_total"))
        )
        .otherwise(None)  # Use None instead of inf for JSON serialization
        .alias("relative_risk"),
        # Log Ratio - log of relative frequencies
        pl.when((pl.col("freq_corpus_0") > 0) & (pl.col("freq_corpus_1") > 0))
        .then(
            (
                (pl.col("freq_corpus_0") / pl.col("corpus_0_total"))
                / (pl.col("freq_corpus_1") / pl.col("corpus_1_total"))
            ).log()
        )
        .otherwise(None)  # Use None instead of 0.0 for consistency
        .alias("log_ratio"),
        # Odds Ratio - odds of occurrence in corpus 1 vs corpus 2
        pl.when(
            (pl.col("freq_corpus_0") > 0)
            & (pl.col("freq_corpus_1") > 0)
            & (pl.col("corpus_1_total") > pl.col("freq_corpus_1"))
            & (pl.col("corpus_0_total") > pl.col("freq_corpus_0"))
        )
        .then(
            (
                pl.col("freq_corpus_0")
                * (pl.col("corpus_1_total") - pl.col("freq_corpus_1"))
            )
            / (
                pl.col("freq_corpus_1")
                * (pl.col("corpus_0_total") - pl.col("freq_corpus_0"))
            )
        )
        .otherwise(None)  # Use None instead of inf for JSON serialization
        .alias("odds_ratio"),
    ])

    return result


def compute_token_frequencies(
    frames, stop_words: Optional[List[str]] = None
) -> tuple[Dict[str, Dict[str, int]], pl.DataFrame]:
    """
    Compute token frequencies and statistical measures across multiple DocDataFrame or DocLazyFrame objects.

    This function tokenizes the document column of each frame and calculates
    token frequencies within each frame, plus log likelihood and effect size statistics.
    All frequency dictionaries share the same set of keys (tokens) for consistent comparison.

    Parameters
    ----------
    frames : Dict[str, DocDataFrame or DocLazyFrame]
        Dictionary mapping frame names to DocDataFrame or DocLazyFrame objects to analyze.
        The keys will be used as names in the returned frequency dictionaries.
    stop_words : List[str], optional
        List of stop words to exclude from frequency calculation.
        If None, no stop words are filtered.

    Returns
    -------
    tuple[Dict[str, Dict[str, int]], pl.DataFrame]
        Tuple containing:
        1. Dictionary mapping frame names to frequency dictionaries.
           Each frequency dictionary maps tokens to their frequency counts within that frame.
           All frequency dictionaries have the same set of keys (union of all tokens).
        2. Polars DataFrame containing statistical measures with columns:
           - token: The token/word
           - log_likelihood_llv: Log likelihood G2 statistic
           - bayes_factor_bic: Bayes factor (BIC)
           - effect_size_ell: Effect size for log likelihood (ELL)
           - significance: Significance level indicator (*** p<0.001, ** p<0.01, * p<0.05)

    Examples
    --------
    >>> import docframe as dp
    >>> df1 = dp.DocDataFrame({"text": ["hello world", "hello there"]})
    >>> df2 = dp.DocDataFrame({"text": ["world peace", "hello world"]})
    >>> frames = {"frame1": df1, "frame2": df2}
    >>> frequencies, stats = dp.compute_token_frequencies(frames)
    >>> list(frequencies.keys())  # Frame names
    ['frame1', 'frame2']
    >>> sorted(frequencies['frame1'].keys())  # Same keys in both
    ['hello', 'peace', 'there', 'world']
    >>> frequencies['frame1']['hello']  # Count in first frame
    2
    >>> frequencies['frame2']['hello']  # Count in second frame
    1
    >>> stats.columns.tolist()  # Statistical measures
    ['token', 'freq_corpus_0', 'freq_corpus_1', 'expected_0', 'expected_1', 'corpus_0_total', 'corpus_1_total', 'log_likelihood_llv', 'bayes_factor_bic', 'effect_size_ell', 'significance', 'percent_corpus_0', 'percent_corpus_1', 'percent_diff', 'relative_risk', 'log_ratio', 'odds_ratio']

    >>> # With stop words
    >>> frequencies, stats = dp.compute_token_frequencies(frames, stop_words=['hello'])
    >>> 'hello' in frequencies['frame1']  # hello is excluded
    False

    Notes
    -----
    - Uses the document column of each frame for tokenization
    - For DocLazyFrame objects, collects them for processing
    - Empty tokens are ignored
    - Case-sensitive tokenization (tokens are lowercased)
    - Tokens are split on whitespace and punctuation
    - Stop words are filtered out before frequency calculation
    - Statistical measures require exactly 2 frames for comparison
    - Log likelihood follows the formula from Rayson & Garside (2000)
    - Effect sizes follow Johnston et al. (2006) and Wilson (2013)
    """
    if not frames:
        raise ValueError("At least one frame must be provided")

    # Import here to avoid circular imports
    from .docframe import DocDataFrame, DocLazyFrame

    # Validate input types
    for name, frame in frames.items():
        if not isinstance(frame, (DocDataFrame, DocLazyFrame)):
            raise TypeError(
                f"Frame '{name}' must be DocDataFrame or DocLazyFrame, got {type(frame)}"
            )

    # Prepare stop words set
    stop_words_set = set(stop_words) if stop_words else set()

    # Collect all tokens from all frames to get the universal vocabulary
    all_tokens = set()
    frame_tokens_lists = {}

    for name, frame in frames.items():
        # Get the document column and tokenize
        if isinstance(frame, DocLazyFrame):
            # For lazy frames, collect first
            doc_series = frame.collect().document
        else:
            doc_series = frame.document

        # Tokenize all documents and flatten
        tokens_list = []
        # Use the text namespace for tokenization
        try:
            tokenized_series = doc_series.text.tokenize()
            for tokens in tokenized_series.to_list():
                if tokens:  # Skip empty token lists
                    # Filter out stop words
                    filtered_tokens = [
                        token for token in tokens if token not in stop_words_set
                    ]
                    tokens_list.extend(filtered_tokens)
                    all_tokens.update(filtered_tokens)
        except Exception:
            # Fallback if text namespace is not available
            for text in doc_series.to_list():
                if text and isinstance(text, str):
                    tokens = simple_tokenize(text)
                    if tokens:
                        filtered_tokens = [
                            token for token in tokens if token not in stop_words_set
                        ]
                        tokens_list.extend(filtered_tokens)
                        all_tokens.update(filtered_tokens)

        frame_tokens_lists[name] = tokens_list

    # Create frequency dictionaries with consistent keys
    result = {}
    freq_dicts_list = []

    for name, tokens_list in frame_tokens_lists.items():
        # Count tokens in this frame
        freq_dict = {}
        for token in tokens_list:
            freq_dict[token] = freq_dict.get(token, 0) + 1

        # Ensure all tokens are represented (with 0 for missing tokens)
        complete_freq_dict = {
            token: freq_dict.get(token, 0) for token in sorted(all_tokens)
        }
        result[name] = complete_freq_dict

        # Store frequency dictionary for statistical calculations
        freq_dicts_list.append(complete_freq_dict)

    # Calculate statistical measures if we have exactly 2 frames
    if len(freq_dicts_list) == 2:
        try:
            stats = _calculate_log_likelihood_and_effect_size(freq_dicts_list)
        except Exception:
            # If statistical calculation fails, create empty stats DataFrame with all required columns
            stats_data = []
            for token in sorted(all_tokens):
                stats_data.append({
                    "token": token,
                    "freq_corpus_0": 0,
                    "freq_corpus_1": 0,
                    "expected_0": 0.0,
                    "expected_1": 0.0,
                    "corpus_0_total": 0,
                    "corpus_1_total": 0,
                    "percent_corpus_0": 0.0,
                    "percent_corpus_1": 0.0,
                    "percent_diff": 0.0,
                    "log_likelihood_llv": 0.0,
                    "bayes_factor_bic": 0.0,
                    "effect_size_ell": 0.0,
                    "relative_risk": None,
                    "log_ratio": None,
                    "odds_ratio": None,
                    "significance": "",
                })
            stats = pl.DataFrame(stats_data)
    else:
        # Create empty stats DataFrame for non-comparison cases with all required columns
        stats_data = []
        for token in sorted(all_tokens):
            stats_data.append({
                "token": token,
                "freq_corpus_0": 0,
                "freq_corpus_1": 0,
                "expected_0": 0.0,
                "expected_1": 0.0,
                "corpus_0_total": 0,
                "corpus_1_total": 0,
                "percent_corpus_0": 0.0,
                "percent_corpus_1": 0.0,
                "percent_diff": 0.0,
                "log_likelihood_llv": 0.0,
                "bayes_factor_bic": 0.0,
                "effect_size_ell": 0.0,
                "relative_risk": None,
                "log_ratio": None,
                "odds_ratio": None,
                "significance": "",
            })
        stats = pl.DataFrame(stats_data)

    return result, stats


def topic_visualization(
    corpora: List[List[str]],
    min_topic_size: int = 5,
    use_ctfidf: bool = False,
    custom_labels: Union[bool, str] = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fit a BERTopic model on multiple corpora and return data needed for a frontend
    inter-topic distance visualization (no Plotly objects created here).

    Per-topic size semantics (UPDATED):
      For each topic we now return `size` as a list whose i-th element is the
      number of documents assigned to that topic coming from the i-th corpus
      in the input `corpora` list. Example: two corpora -> size = [N_from_corpus0, M_from_corpus1].

    Returns dict with:
      - corpus_sizes: list[int]
      - topics: list[ { id, label, size: List[int], total_size, x, y } ]
      - per_corpus_topic_counts: list[dict]
      - assignments: list[list[int]]
      - meta: auxiliary info
    """
    if not corpora or any(c is None for c in corpora):
        raise ValueError("'corpora' must be a non-empty list of document lists.")

    corpus_sizes = [len(c) for c in corpora]
    if any(sz == 0 for sz in corpus_sizes):
        raise ValueError("All corpora must contain at least one document.")

    # Merge corpora for fitting
    full_list = reduce(lambda x, y: x + y, corpora)

    if not _HAS_BERTOPIC:
        raise ImportError(
            "BERTopic is required for topic_visualization but is not installed."
        )

    # Local imports to minimize import cost for users not using BERTopic functions
    from bertopic import BERTopic  # type: ignore
    from bertopic._utils import select_topic_representation  # type: ignore

    # Fit model (silence verbose)
    topic_model = BERTopic(min_topic_size=min_topic_size, verbose=False)
    topic_model.fit(full_list)

    # Transform each corpus separately (returns (topics, probs))
    transformed = [topic_model.transform(corpus) for corpus in corpora]
    assignments = [t[0] for t in transformed]

    # Build per-corpus topic count dictionaries
    per_corpus_topic_counts: List[Dict[int, int]] = []
    for topic_ids in assignments:
        counts: Dict[int, int] = {}
        for tid in topic_ids:
            counts[tid] = counts.get(tid, 0) + 1
        per_corpus_topic_counts.append(counts)

    # Prepare topic list excluding outlier (-1)
    freq_df = topic_model.get_topic_freq()
    topic_ids = [int(t) for t in freq_df[freq_df.Topic != -1].Topic.tolist()]

    # Topic labels
    if isinstance(custom_labels, str):  # aspect-based labels
        words_nested = [
            [[str(t), None]] + topic_model.topic_aspects_[custom_labels][t]
            for t in topic_ids
        ]
        labels = ["_".join([w[0] for w in wn[:4]]) for wn in words_nested]
        labels = [lbl if len(lbl) < 30 else lbl[:27] + "..." for lbl in labels]
    elif custom_labels and getattr(topic_model, "custom_labels_", None) is not None:
        labels = [
            topic_model.custom_labels_[t + topic_model._outliers] for t in topic_ids
        ]
    else:
        labels = [
            " | ".join([w[0] for w in topic_model.get_topic(t)[:5]]) for t in topic_ids
        ]

    # Extract base embeddings (topic embeddings or c-TF-IDF)
    all_topics_sorted = sorted(list(topic_model.get_topics().keys()))
    indices = (
        np.array([all_topics_sorted.index(t) for t in topic_ids])
        if topic_ids
        else np.array([])
    )

    embeddings, c_tfidf_used = select_topic_representation(  # type: ignore
        topic_model.c_tf_idf_,
        topic_model.topic_embeddings_,
        use_ctfidf=use_ctfidf,
        output_ndarray=True,
    )
    if len(indices) > 0:
        embeddings = embeddings[indices]
    else:
        embeddings = np.zeros((0, 2))

    # Dimensionality reduction to 2D
    coords: np.ndarray
    if embeddings.shape[0] == 0:
        coords = embeddings
    elif embeddings.shape[0] == 1:  # Single topic
        coords = np.array([[0.0, 0.0]])
    else:
        if _HAS_UMAP:
            if c_tfidf_used:
                emb_norm = MinMaxScaler().fit_transform(embeddings)
                coords = UMAP(
                    n_neighbors=2,
                    n_components=2,
                    metric="hellinger",
                    random_state=random_state,
                ).fit_transform(emb_norm)
            else:
                coords = UMAP(
                    n_neighbors=2,
                    n_components=2,
                    metric="cosine",
                    random_state=random_state,
                ).fit_transform(embeddings)
        else:  # Fallback PCA projection for deterministic tests without umap
            from sklearn.decomposition import PCA

            comps = min(2, embeddings.shape[1])
            proj = PCA(n_components=comps, random_state=random_state).fit_transform(
                embeddings
            )
            if comps == 1:
                coords = np.column_stack([proj[:, 0], np.zeros_like(proj[:, 0])])
            else:
                coords = proj

    # Assemble topic data with per-corpus size list
    topics_payload = []
    for i, tid in enumerate(topic_ids):
        per_corpus_sizes = [
            per_corpus_topic_counts[j].get(tid, 0)
            for j in range(len(per_corpus_topic_counts))
        ]
        topics_payload.append({
            "id": tid,
            "label": labels[i],
            # size now list aligned with corpora order
            "size": per_corpus_sizes,
            "total_size": int(sum(per_corpus_sizes)),
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
        })

    return {
        "corpus_sizes": corpus_sizes,
        "topics": topics_payload,
        "per_corpus_topic_counts": per_corpus_topic_counts,
        "assignments": assignments,
        "meta": {
            "used_ctfidf": bool(use_ctfidf),
            "embeddings_from_ctfidf": bool(c_tfidf_used),
            "min_topic_size": min_topic_size,
            "total_topics_incl_outlier": int(freq_df.shape[0]),
        },
    }
