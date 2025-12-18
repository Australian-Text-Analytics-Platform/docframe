"""
Text processing utilities
"""

import os
import re
import string
import warnings
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import nltk
import numpy as np
import polars as pl
from nltk.tokenize import TreebankWordDetokenizer, TreebankWordTokenizer, word_tokenize
from sklearn.preprocessing import MinMaxScaler

# Fix NLTK path handling on Windows to prevent mixed separator issues
# Wrap in try-except to prevent import-time crashes
if os.name == "nt":  # Windows
    import sys
    print("[text_utils] Starting Windows NLTK path patching...", file=sys.stderr, flush=True)
    try:
        from pathlib import Path as _Path
        print("[text_utils] Imported Path successfully", file=sys.stderr, flush=True)
        
        # Store original functions
        _original_nltk_find = nltk.data.find
        _original_nltk_load = nltk.data.load
        print("[text_utils] Stored original NLTK functions", file=sys.stderr, flush=True)
        
        def _normalize_path_windows(path_str):
            """Normalize a path string for Windows, removing UNC prefixes and mixed separators."""
            try:
                if not path_str or not isinstance(path_str, str):
                    return path_str
                
                # Replace forward slashes with backslashes for Windows
                normalized = path_str.replace("/", "\\")
                
                # Use os.path.normpath to clean up the path without filesystem access
                normalized = os.path.normpath(normalized)
                
                return normalized
            except Exception:
                # If normalization fails, return original
                return path_str
        
        def _patched_nltk_find(resource_name, paths=None):
            """Patched NLTK find that normalizes paths on Windows."""
            try:
                result = _original_nltk_find(resource_name, paths)
                if isinstance(result, str):
                    return _normalize_path_windows(result)
                return result
            except Exception as e:
                # If patching fails, fall back to original behavior
                raise
        
        def _patched_nltk_load(resource_url, format='auto', cache=True, verbose=False, 
                               logic_parser=None, fstruct_reader=None, encoding=None):
            """Patched NLTK load that normalizes file paths on Windows."""
            try:
                # Normalize the resource_url if it's a file path
                if isinstance(resource_url, str) and not resource_url.startswith(('http:', 'https:', 'file:')):
                    resource_url = _normalize_path_windows(resource_url)
            except Exception:
                # If normalization fails, use original path
                pass
            
            return _original_nltk_load(
                resource_url, format=format, cache=cache, verbose=verbose,
                logic_parser=logic_parser, fstruct_reader=fstruct_reader, encoding=encoding
            )
        
        # Apply patches only if everything succeeded
        nltk.data.find = _patched_nltk_find
        nltk.data.load = _patched_nltk_load
        print("[text_utils] SUCCESS: Windows NLTK path patches applied successfully", file=sys.stderr, flush=True)
        
    except Exception as e:
        # If Windows patching fails entirely, print warning but don't crash
        import sys
        import traceback
        print(f"[text_utils] ERROR: Failed to apply NLTK Windows path patches: {e}", file=sys.stderr, flush=True)
        print(f"[text_utils] Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        # Continue without patches - NLTK will use default behavior
else:
    import sys
    print("[text_utils] Not Windows, skipping NLTK patching", file=sys.stderr, flush=True)

# Suppress sklearn deprecation warnings
warnings.filterwarnings(
    "ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'"
)

try:  # Lazy optional BERTopic dependency (avoid heavy import at module load)
    import importlib.util

    _HAS_BERTOPIC = importlib.util.find_spec("bertopic") is not None
except Exception:  # pragma: no cover
    _HAS_BERTOPIC = False

# Detect UMAP availability without importing it (to avoid heavy deps at import time)
try:  # Optional dependency: if unavailable, we fall back to PCA
    import importlib.util as _il_util

    _HAS_UMAP = _il_util.find_spec("umap") is not None
except Exception:  # pragma: no cover - environment without umap
    _HAS_UMAP = False
UMAP = None  # type: ignore

_NLTK_PUNKT_READY = False
_NLTK_POS_READY = False
_DETOKENIZER = TreebankWordDetokenizer()
_TREEBANK_TOKENIZER = TreebankWordTokenizer()
_PUNKT_SENTENCE_TOKENIZER = None


def _ensure_nltk_punkt() -> None:
    global _NLTK_PUNKT_READY
    if _NLTK_PUNKT_READY:
        return
    try:
        nltk.data.find("tokenizers/punkt")
        _NLTK_PUNKT_READY = True
    except LookupError:
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            # punkt_tab is optional; ignore download failures to remain offline-friendly
            pass
        _NLTK_PUNKT_READY = True


def _ensure_nltk_pos_tagger() -> None:
    global _NLTK_POS_READY
    if _NLTK_POS_READY:
        return
    tagger_paths = [
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng",
    ]
    found_any = False
    for path in tagger_paths:
        try:
            nltk.data.find(path)
            found_any = True
        except LookupError:
            resource_name = path.split("/")[-1]
            try:
                nltk.download(resource_name, quiet=True)
                nltk.data.find(path)
                found_any = True
            except Exception:
                # Some environments provide only one of the taggers; continue trying others
                continue
    if found_any:
        _NLTK_POS_READY = True


def _get_sentence_tokenizer():
    global _PUNKT_SENTENCE_TOKENIZER
    if _PUNKT_SENTENCE_TOKENIZER is None:
        _ensure_nltk_punkt()
        try:
            _PUNKT_SENTENCE_TOKENIZER = nltk.data.load(
                "tokenizers/punkt/english.pickle"
            )
        except LookupError:
            # Fallback: instantiate a default PunktSentenceTokenizer
            _PUNKT_SENTENCE_TOKENIZER = nltk.tokenize.PunktSentenceTokenizer()
    return _PUNKT_SENTENCE_TOKENIZER


def tokenize(text: str, lowercase: bool = True, remove_punct: bool = True) -> List[str]:
    """
    Tokenize text using NLTK's word_tokenize with optional normalization.

    Wraps NLTK's word_tokenize function with preprocessing options. Automatically downloads
    required NLTK data (punkt tokenizer) if not already available.

    Args:
        text: Input text string to tokenize. Must be a string type.
        lowercase: If True, convert text to lowercase before tokenization. Defaults to True.
        remove_punct: If True, filter out tokens containing only punctuation.
            Keeps tokens with at least one alphanumeric character. Defaults to True.

    Returns:
        List[str]: List of tokens. Empty list if input is empty or contains no tokens.

    Raises:
        TypeError: If input text is not a string.

    Examples:
        >>> tokenize("Hello, World!")
        ['hello', ',', 'world', '!']  # with remove_punct=False

        >>> tokenize("Hello, World!", remove_punct=True)
        ['hello', 'world']

        >>> tokenize("It's a test.", lowercase=False, remove_punct=True)
        ["It's", 'a', 'test']

    Notes:
        - Uses NLTK's TreebankWordTokenizer internally
        - Automatically downloads 'punkt' and 'punkt_tab' if needed
        - Punctuation removal checks for any alphanumeric character (isalnum)
        - Empty strings or None values raise TypeError
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    processed_text = text.lower() if lowercase else text

    _ensure_nltk_punkt()

    try:
        tokens = word_tokenize(processed_text)
    except (OSError, IOError) as e:
        # Fallback to simple whitespace tokenization if NLTK data loading fails
        # This can happen on Windows with path issues
        import warnings
        warnings.warn(
            f"NLTK tokenization failed ({e}), falling back to simple split. "
            "This may affect tokenization quality.",
            RuntimeWarning
        )
        # Simple fallback: split on whitespace and basic punctuation
        import re
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", processed_text)

    if remove_punct:
        tokens = [tok for tok in tokens if any(ch.isalnum() for ch in tok)]

    return tokens


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_digits: bool = False,
    remove_extra_whitespace: bool = True,
) -> str:
    """
    Clean text with various normalization options.

    Applies multiple text cleaning operations in sequence for preprocessing pipelines.
    Returns empty string for non-string inputs instead of raising errors.

    Args:
        text: Input text string to clean. Non-string inputs return empty string.
        lowercase: If True, convert text to lowercase. Defaults to True.
        remove_punct: If True, remove all punctuation characters. Defaults to True.
        remove_digits: If True, remove all digit characters. Defaults to False.
        remove_extra_whitespace: If True, collapse multiple whitespace characters
            into single spaces and strip leading/trailing whitespace. Defaults to True.

    Returns:
        str: Cleaned text string. Empty string if input is not a string.

    Examples:
        >>> clean_text("  Hello,  World!  123  ")
        'hello world 123'  # lowercase, punct removed, whitespace normalized

        >>> clean_text("Price: $50.99", remove_digits=True)
        'price '  # digits removed

        >>> clean_text("Test!", lowercase=False, remove_punct=False)
        'Test!'  # no modifications

    Notes:
        - Operations applied in order: lowercase → remove_punct → remove_digits → whitespace
        - Uses str.translate for punctuation removal (string.punctuation)
        - Uses regex r"\\d+" for digit removal
        - Uses regex r"\\s+" for whitespace normalization
        - Gracefully handles non-string inputs by returning empty string
    """
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
    """
    Count words in text using whitespace splitting.

    Simple word count using str.split() without arguments (splits on any whitespace).
    Returns 0 for non-string or empty inputs.

    Args:
        text: Input text string. Non-string inputs return 0.

    Returns:
        int: Number of whitespace-separated tokens. 0 for empty or non-string inputs.

    Examples:
        >>> word_count("Hello world")
        2
        >>> word_count("  Multiple   spaces  ")
        2
        >>> word_count("")
        0

    Notes:
        - Does not handle punctuation or special tokenization
        - Consecutive whitespace treated as single separator
        - For more accurate word counting, use tokenize() instead
    """
    if not isinstance(text, str):
        return 0
    return len(text.split())


def char_count(text: str) -> int:
    """
    Count total characters in text including spaces and punctuation.

    Uses len() to count all characters. Returns 0 for non-string inputs.

    Args:
        text: Input text string. Non-string inputs return 0.

    Returns:
        int: Total character count including whitespace and punctuation. 0 for non-string.

    Examples:
        >>> char_count("Hello")
        5
        >>> char_count("Hi there!")
        9  # includes space and punctuation
        >>> char_count("")
        0
    """
    if not isinstance(text, str):
        return 0
    return len(text)


def sentence_count(text: str) -> int:
    """
    Count sentences in text using simple regex splitting.

    Splits text on common sentence-ending punctuation [.!?]+ and counts non-empty segments.
    Returns 0 for non-string inputs.

    Args:
        text: Input text string. Non-string inputs return 0.

    Returns:
        int: Number of detected sentences. Empty or whitespace-only segments ignored.

    Examples:
        >>> sentence_count("Hello! How are you?")
        2
        >>> sentence_count("One sentence.")
        1
        >>> sentence_count("Multiple!!! Exclamation!!! Marks!!!")
        3

    Notes:
        - Uses regex r"[.!?]+" for splitting (handles multiple punctuation marks)
        - Does not handle abbreviations (e.g., "Dr.", "U.S.") correctly
        - Does not handle quotation marks or complex punctuation
        - For more accurate sentence segmentation, consider NLTK's sentence tokenizer
    """
    if not isinstance(text, str):
        return 0
    # Simple sentence splitting on common sentence endings
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """
    Extract n-grams from text as space-separated token sequences.

    Tokenizes text and creates sliding windows of n consecutive tokens.
    Returns empty list for non-string inputs or if text has fewer than n tokens.

    Args:
        text: Input text string to extract n-grams from. Non-string inputs return [].
        n: Size of n-grams (number of consecutive tokens). Defaults to 2 (bigrams).

    Returns:
        List[str]: List of n-grams as space-separated strings. Empty if insufficient tokens.

    Examples:
        >>> extract_ngrams("the quick brown fox", n=2)
        ['the quick', 'quick brown', 'brown fox']

        >>> extract_ngrams("the quick brown fox", n=3)
        ['the quick brown', 'quick brown fox']

        >>> extract_ngrams("short", n=2)
        []  # only 1 token, need at least 2

    Notes:
        - Uses tokenize() internally (lowercase=True, remove_punct=True by default)
        - N-grams are space-separated strings, not tuples
        - Sliding window approach: [0:n], [1:n+1], ..., [len-n:len]
        - Returns empty list if len(tokens) < n
    """
    if not isinstance(text, str):
        return []

    tokens = tokenize(text)
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


# -----------------------------
# Quotation extraction (heuristic)
# -----------------------------


class _TokenInfo(NamedTuple):
    text: str
    start: int
    end: int
    pos: str
    sentence_index: int


class _SentenceInfo(NamedTuple):
    index: int
    start: int
    end: int
    text: str


@dataclass
class _QuoteRecord:
    speaker: str
    speaker_start_idx: Optional[int]
    speaker_end_idx: Optional[int]
    quote: str
    quote_start_idx: int
    quote_end_idx: int
    verb: str
    verb_start_idx: Optional[int]
    verb_end_idx: Optional[int]
    quote_type: str
    quote_token_count: int
    is_floating_quote: bool
    sentence_index: int

    def to_public_dict(self) -> Dict[str, Any]:
        def _idx_or_none(value: int) -> Optional[int]:
            return None if value is None or value < 0 else int(value)

        def _text_or_none(value: str) -> Optional[str]:
            value = value or ""
            return value if value.strip() else None

        return {
            "speaker": _text_or_none(self.speaker),
            "speaker_start_idx": _idx_or_none(self.speaker_start_idx),
            "speaker_end_idx": _idx_or_none(self.speaker_end_idx),
            "quote": self.quote,
            "quote_start_idx": int(self.quote_start_idx),
            "quote_end_idx": int(self.quote_end_idx),
            "verb": _text_or_none(self.verb),
            "verb_start_idx": _idx_or_none(self.verb_start_idx),
            "verb_end_idx": _idx_or_none(self.verb_end_idx),
            "quote_type": self.quote_type,
            "quote_token_count": int(self.quote_token_count),
            "is_floating_quote": bool(self.is_floating_quote),
        }


_TITLE_WORDS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sir",
    "madam",
    "president",
    "minister",
    "senator",
    "sen",
    "rep",
    "capt",
    "gen",
    "gov",
    "amb",
    "lord",
    "lady",
}

_BOUNDARY_TOKENS = {",", ";", ":", "-", "—", "–", ".", "!", "?"}
_ALLOWED_PRONOUN_SPEAKERS = {"he", "she", "they", "him", "her", "them"}
_INVALID_SPEAKER_WORDS = {"i", "we"}
_QUOTE_CHAR_PATTERN = re.compile(r"[\"“”«»„‟]")
_QUOTE_PAIR_THRESHOLD = 200
_VERB_WINDOW_CHARS = 200
_SPEAKER_HOP_LIMIT = 60
_MIN_QUOTE_TOKEN_COUNT = 3


def _prepare_tokens_and_sentences(
    text: str,
) -> Tuple[List[_TokenInfo], List[_SentenceInfo]]:
    _ensure_nltk_punkt()
    _ensure_nltk_pos_tagger()

    sentence_tokenizer = _get_sentence_tokenizer()
    tokens: List[_TokenInfo] = []
    sentences: List[_SentenceInfo] = []

    for idx, (start, end) in enumerate(sentence_tokenizer.span_tokenize(text)):
        sent_text = text[start:end]
        spans = list(_TREEBANK_TOKENIZER.span_tokenize(sent_text))
        words = [sent_text[s:e] for s, e in spans]
        if words:
            try:
                pos_tags = [tag for _, tag in nltk.pos_tag(words)]
            except LookupError:
                # Retry after forcing tagger availability (handles environments missing default models)
                global _NLTK_POS_READY
                _NLTK_POS_READY = False
                _ensure_nltk_pos_tagger()
                pos_tags = [tag for _, tag in nltk.pos_tag(words)]
        else:
            pos_tags = []

        for (tok_start, tok_end), word, pos in zip(spans, words, pos_tags):
            tokens.append(
                _TokenInfo(
                    text=word,
                    start=start + tok_start,
                    end=start + tok_end,
                    pos=pos,
                    sentence_index=idx,
                )
            )

        sentences.append(_SentenceInfo(index=idx, start=start, end=end, text=sent_text))

    return tokens, sentences


def _detect_quote_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    open_idx: Optional[int] = None
    for match in _QUOTE_CHAR_PATTERN.finditer(text):
        if open_idx is None:
            open_idx = match.start()
        else:
            spans.append((open_idx, match.end()))
            open_idx = None
    return spans


def _count_tokens_in_span(span_text: str) -> int:
    return sum(1 for _ in _TREEBANK_TOKENIZER.span_tokenize(span_text))


def _token_distance(start: int, end: int, span_start: int, span_end: int) -> int:
    if end <= span_start:
        return span_start - end
    if start >= span_end:
        return start - span_end
    return 0


def _find_nearest_verb(
    tokens: List[_TokenInfo],
    verbs: set[str],
    span_start: int,
    span_end: int,
    target_sentence_idx: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    window_start = max(0, span_start - _VERB_WINDOW_CHARS)
    window_end = span_end + _VERB_WINDOW_CHARS
    candidates: List[Tuple[int, int, int, str, int, int]] = []

    for idx, token in enumerate(tokens):
        if token.end < window_start:
            continue
        if token.start > window_end:
            break

        if (
            target_sentence_idx is not None
            and token.sentence_index != target_sentence_idx
        ):
            continue

        # Skip tokens that overlap the quote span; we only want context verbs
        if not (token.end <= span_start or token.start >= span_end):
            continue

        lower = token.text.lower()

        if lower == "according" and (idx + 1) < len(tokens):
            next_token = tokens[idx + 1]
            if (
                target_sentence_idx is not None
                and next_token.sentence_index != target_sentence_idx
            ):
                continue
            if not (next_token.end <= span_start or next_token.start >= span_end):
                continue
            if next_token.text.lower() == "to":
                vs = token.start
                ve = next_token.end
                dist = _token_distance(vs, ve, span_start, span_end)
                candidates.append((dist, vs, ve, "according to", idx, idx + 1))
                continue

        # Ignore non-alphabetic tokens that POS tagging mislabels as verbs
        has_alpha = any(ch.isalpha() for ch in token.text)
        if not has_alpha:
            continue

        is_candidate = lower in verbs or token.pos.startswith("VB")
        if lower in {"is", "was", "be"}:
            is_candidate = False
        if not is_candidate:
            continue

        vs, ve = token.start, token.end
        dist = _token_distance(vs, ve, span_start, span_end)
        candidates.append((dist, vs, ve, token.text, idx, idx))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    dist, vs, ve, verb_text, token_idx, token_end_idx = candidates[0]
    return {
        "verb": verb_text,
        "verb_start": vs,
        "verb_end": ve,
        "token_index": token_idx,
        "token_end_index": token_end_idx,
        "distance": dist,
    }


def _is_name_token(token: _TokenInfo) -> bool:
    text = token.text
    if not text:
        return False
    stripped = text.rstrip(".")
    if token.pos in {"NNP", "NNPS"}:
        return True
    if stripped and stripped[0].isupper() and any(c.isalpha() for c in stripped):
        return True
    if len(stripped) > 1 and stripped.isupper() and stripped.isalpha():
        return True
    if stripped.lower() in _TITLE_WORDS:
        return True
    return False


def _is_pronoun_token(token: _TokenInfo) -> bool:
    lower = token.text.lower()
    return token.pos in {"PRP", "PRP$"} and lower not in _INVALID_SPEAKER_WORDS


def _find_speaker_near(
    text: str,
    tokens: List[_TokenInfo],
    verb_index: int,
    verb_end_index: int,
    direction: str,
) -> Tuple[str, Optional[int], Optional[int]]:
    if direction == "left":
        idx = verb_index - 1
        hops = 0
        while idx >= 0 and hops < _SPEAKER_HOP_LIMIT:
            token = tokens[idx]
            if token.text in _BOUNDARY_TOKENS:
                break
            if _is_name_token(token):
                end_idx = idx
                while end_idx + 1 < len(tokens) and _is_name_token(tokens[end_idx + 1]):
                    end_idx += 1
                start_idx = idx
                while start_idx - 1 >= 0 and _is_name_token(tokens[start_idx - 1]):
                    start_idx -= 1
                start = tokens[start_idx].start
                end = tokens[end_idx].end
                return text[start:end], start, end
            if _is_pronoun_token(token):
                return text[token.start : token.end], token.start, token.end
            idx -= 1
            hops += 1
    else:
        idx = verb_end_index + 1
        hops = 0
        while idx < len(tokens) and hops < _SPEAKER_HOP_LIMIT:
            token = tokens[idx]
            if token.text in _BOUNDARY_TOKENS:
                break
            if _is_name_token(token):
                end_idx = idx
                while end_idx + 1 < len(tokens) and _is_name_token(tokens[end_idx + 1]):
                    end_idx += 1
                start = tokens[idx].start
                end = tokens[end_idx].end
                return text[start:end], start, end
            if _is_pronoun_token(token):
                return text[token.start : token.end], token.start, token.end
            idx += 1
            hops += 1
    return "", None, None


def _compute_quote_type(
    quote_start: int,
    quote_end: int,
    verb_start: Optional[int],
    verb_end: Optional[int],
    speaker_start: Optional[int],
    speaker_end: Optional[int],
) -> str:
    verb_start = -1 if verb_start is None else verb_start
    verb_end = -1 if verb_end is None else verb_end
    speaker_start = -1 if speaker_start is None else speaker_start
    speaker_end = -1 if speaker_end is None else speaker_end
    positions: List[Tuple[str, float]] = []
    if quote_start >= 0:
        positions.append(("Q", float(quote_start)))
    content_mid = (
        (quote_start + quote_end) / 2 if quote_end > quote_start else quote_start
    )
    positions.append(("C", float(content_mid)))
    if quote_end >= 0:
        positions.append(("q", float(quote_end)))
    if verb_start >= 0 and verb_end >= 0:
        positions.append(("V", float((verb_start + verb_end) / 2)))
    if speaker_start >= 0 and speaker_end >= 0:
        positions.append(("S", float((speaker_start + speaker_end) / 2)))
    positions.sort(key=lambda x: (x[1], x[0]))
    return "".join(code for code, _ in positions).replace("q", "Q")


def _inherit_floating_quotes(
    records: List[_QuoteRecord],
    sentences: List[_SentenceInfo],
) -> List[_QuoteRecord]:
    if not records:
        return records

    records_sorted = sorted(records, key=lambda r: r.quote_start_idx)
    last_structured: Optional[_QuoteRecord] = None

    for record in records_sorted:
        sentence_idx = record.sentence_index
        sentence = (
            sentences[sentence_idx] if 0 <= sentence_idx < len(sentences) else None
        )
        sentence_starts_with_quote = False
        if sentence:
            stripped = sentence.text.lstrip()
            sentence_starts_with_quote = stripped.startswith(
                '"'
            ) or stripped.startswith("“")

        can_inherit = (
            last_structured is not None
            and bool(last_structured.speaker)
            and sentence_idx - last_structured.sentence_index <= 5
            and not record.speaker
            and not record.verb
            and sentence_starts_with_quote
        )

        if can_inherit:
            record.speaker = last_structured.speaker
            record.speaker_start_idx = last_structured.speaker_start_idx
            record.speaker_end_idx = last_structured.speaker_end_idx
            record.is_floating_quote = True

        if record.verb or record.speaker:
            last_structured = record

    return records_sorted


def _deduplicate_quotes(records: List[_QuoteRecord]) -> List[_QuoteRecord]:
    if not records:
        return []

    sorted_by_span = sorted(
        records,
        key=lambda r: (r.quote_token_count, r.quote_end_idx - r.quote_start_idx),
        reverse=True,
    )
    kept: List[_QuoteRecord] = []
    spans: List[Tuple[int, int]] = []

    for record in sorted_by_span:
        start, end = record.quote_start_idx, record.quote_end_idx
        overlap = any(not (end <= s or start >= e) for s, e in spans)
        if overlap:
            continue
        if record.quote_token_count < _MIN_QUOTE_TOKEN_COUNT and len(records) > 1:
            continue
        spans.append((start, end))
        kept.append(record)

    kept.sort(key=lambda r: r.quote_start_idx)
    return kept


def quotation_elements(text: Optional[str]) -> List[Dict[str, Any]]:
    """Extract quotations from a single text using NLP-lite heuristics."""

    if not isinstance(text, str) or not text:
        return []

    def _load_quote_verbs_inner() -> set[str]:
        candidates: List[str] = []
        try:
            this_dir = os.path.dirname(__file__)
            data_path = os.path.normpath(
                os.path.join(this_dir, "..", "data", "quote_verb.txt")
            )
            if os.path.exists(data_path):
                with open(data_path, "r", encoding="utf-8") as f:
                    candidates.extend([ln.strip() for ln in f if ln.strip()])
        except Exception:
            candidates = []
        if not candidates:
            candidates = [
                "say",
                "said",
                "says",
                "tell",
                "told",
                "writes",
                "wrote",
                "tweet",
                "tweeted",
                "tweeting",
                "according to",
                "state",
                "stated",
                "states",
                "report",
                "reported",
                "reports",
            ]
        return set(c.lower() for c in candidates if c)

    verbs = _load_quote_verbs_inner()

    tokens, sentences = _prepare_tokens_and_sentences(text)
    quote_spans = _detect_quote_spans(text)
    if not quote_spans:
        return []

    records: List[_QuoteRecord] = []

    for qs, qe in quote_spans:
        quote_text = text[qs:qe]
        quote_token_count = _count_tokens_in_span(quote_text)

        sentence_index = 0
        for sent in sentences:
            if sent.start <= qs < sent.end:
                sentence_index = sent.index
                break
            if qs >= sent.end:
                sentence_index = sent.index

        verb_info = _find_nearest_verb(tokens, verbs, qs, qe, sentence_index)
        verb_text = ""
        verb_start = -1
        verb_end = -1
        verb_token_index = -1
        verb_token_end_index = -1

        speaker_text = ""
        speaker_start = -1
        speaker_end = -1

        if verb_info:
            verb_text = verb_info["verb"]
            verb_start = verb_info["verb_start"]
            verb_end = verb_info["verb_end"]
            verb_token_index = verb_info["token_index"]
            verb_token_end_index = verb_info["token_end_index"]

            if verb_end <= qs:
                direction = "left"
            elif verb_start >= qe:
                direction = "right"
            else:
                direction = "left"

            speaker_text, speaker_start, speaker_end = _find_speaker_near(
                text,
                tokens,
                verb_token_index,
                verb_token_end_index,
                direction,
            )
            if not speaker_text:
                fallback_direction = "right" if direction == "left" else "left"
                speaker_text, speaker_start, speaker_end = _find_speaker_near(
                    text,
                    tokens,
                    verb_token_index,
                    verb_token_end_index,
                    fallback_direction,
                )

        if not verb_info:
            window = 200
            left_ctx_start = max(0, qs - window)
            left_ctx = text[left_ctx_start:qs]
            match = re.search(
                r"according to\s+([^,\n]+)", left_ctx, flags=re.IGNORECASE
            )
            if match:
                verb_text = "according to"
                verb_start = left_ctx_start + match.start()
                verb_end = left_ctx_start + match.start() + len("according to")
                speaker_start = left_ctx_start + match.start(1)
                speaker_end = left_ctx_start + match.end(1)
                speaker_text = text[speaker_start:speaker_end]

        speaker_text = (speaker_text or "").strip()
        if speaker_text and speaker_text.lower() in _INVALID_SPEAKER_WORDS:
            speaker_text = ""
            speaker_start = -1
            speaker_end = -1

        if verb_text and verb_text.lower() == "according to":
            quote_type = "AccordingTo"
        else:
            quote_type = _compute_quote_type(
                qs,
                qe,
                verb_start,
                verb_end,
                speaker_start,
                speaker_end,
            )

        record = _QuoteRecord(
            speaker=speaker_text,
            speaker_start_idx=speaker_start,
            speaker_end_idx=speaker_end,
            quote=quote_text,
            quote_start_idx=qs,
            quote_end_idx=qe,
            verb=verb_text,
            verb_start_idx=verb_start,
            verb_end_idx=verb_end,
            quote_type=quote_type,
            quote_token_count=quote_token_count,
            is_floating_quote=False,
            sentence_index=sentence_index,
        )
        records.append(record)

    records = _inherit_floating_quotes(records, sentences)
    records = _deduplicate_quotes(records)

    return [record.to_public_dict() for record in records]


def concordance_elements(
    text: Optional[str],
    search_word: str,
    num_left_tokens: int = 10,
    num_right_tokens: int = 10,
    regex: bool = False,
    case_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """Element-wise concordance: returns matches with contexts and indices for a single text.

    Returns list of dicts with keys:
      - left_context: str
      - matched_text: str
      - right_context: str
      - start_idx: int (match start char index)
      - end_idx: int (match end char index)
      - l1: str (token immediately to the left, or "")
      - r1: str (token immediately to the right, or "")
    """
    if (
        not isinstance(text, str)
        or not isinstance(search_word, str)
        or len(search_word) == 0
    ):
        return []

    pattern = search_word if regex else re.escape(search_word)
    flags = 0 if case_sensitive else re.IGNORECASE
    searcher = re.compile(pattern, flags)

    results: List[Dict[str, Any]] = []
    for match in searcher.finditer(text):
        matched_text = match.group(0)
        start_idx = match.start()
        end_idx = match.end()

        left_text = text[:start_idx]
        right_text = text[end_idx:]

        # Tokenize left/right contexts without altering case or punctuation
        left_tokens = tokenize(left_text, lowercase=False, remove_punct=False)
        right_tokens = tokenize(right_text, lowercase=False, remove_punct=False)

        left_context_tokens = (
            left_tokens[-num_left_tokens:] if num_left_tokens > 0 else []
        )
        right_context_tokens = (
            right_tokens[:num_right_tokens] if num_right_tokens > 0 else []
        )

        l1 = left_context_tokens[-1] if left_context_tokens else ""
        r1 = right_context_tokens[0] if right_context_tokens else ""

        left_context = (
            _DETOKENIZER.detokenize(left_context_tokens) if left_context_tokens else ""
        )
        right_context = (
            _DETOKENIZER.detokenize(right_context_tokens)
            if right_context_tokens
            else ""
        )

        results.append({
            "left_context": left_context,
            "matched_text": matched_text,
            "right_context": right_context,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "l1": l1,
            "r1": r1,
        })

    return results


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
                    tokens = tokenize(text)
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
    elif embeddings.shape[0] <= 15:  # Use PCA for small datasets to avoid UMAP issues
        # Use PCA for small datasets to avoid UMAP scipy.linalg.eigh issues
        from sklearn.decomposition import PCA

        comps = min(2, embeddings.shape[1])
        proj = PCA(n_components=comps, random_state=random_state).fit_transform(
            embeddings
        )
        if comps == 1:
            coords = np.column_stack([proj[:, 0], np.zeros_like(proj[:, 0])])
        else:
            coords = proj
    else:
        if _HAS_UMAP:
            # Lazy import here to avoid numba/llvmlite initialization during module import
            from umap import UMAP  # type: ignore

            # Adjust n_neighbors based on data size to avoid scipy.linalg.eigh issues
            # UMAP needs n_neighbors < n_samples and sufficient samples for embedding
            n_samples = embeddings.shape[0]
            # Use stricter calculation: n_neighbors must be < n_samples - 1
            n_neighbors = max(2, min(15, n_samples - 2))

            try:
                if c_tfidf_used:
                    emb_norm = MinMaxScaler().fit_transform(embeddings)
                    coords = UMAP(
                        n_neighbors=n_neighbors,
                        n_components=2,
                        metric="hellinger",
                        random_state=random_state,
                    ).fit_transform(emb_norm)
                else:
                    coords = UMAP(
                        n_neighbors=n_neighbors,
                        n_components=2,
                        metric="cosine",
                        random_state=random_state,
                    ).fit_transform(embeddings)
            except (TypeError, ValueError, RuntimeError) as e:
                # Fallback to PCA if UMAP fails (including scipy.linalg.eigh errors)
                print(f"UMAP failed with error: {e}. Falling back to PCA.")
                from sklearn.decomposition import PCA

                comps = min(2, embeddings.shape[1])
                proj = PCA(n_components=comps, random_state=random_state).fit_transform(
                    embeddings
                )
                if comps == 1:
                    coords = np.column_stack([proj[:, 0], np.zeros_like(proj[:, 0])])
                else:
                    coords = proj
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
