"""
Text processing utilities
"""
import re
import string
from typing import Dict, List, Optional


def simple_tokenize(text: str, lowercase: bool = True, remove_punct: bool = True) -> List[str]:
    """Simple tokenization using regex"""
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Remove punctuation if requested
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split on whitespace
    tokens = text.split()
    return [token.strip() for token in tokens if token.strip()]


def clean_text(text: str, 
               lowercase: bool = True,
               remove_punct: bool = True,
               remove_digits: bool = False,
               remove_extra_whitespace: bool = True) -> str:
    """Clean text with various options"""
    if not isinstance(text, str):
        return ""
    
    result = text
    
    if lowercase:
        result = result.lower()
    
    if remove_punct:
        result = result.translate(str.maketrans('', '', string.punctuation))
    
    if remove_digits:
        result = re.sub(r'\d+', '', result)
    
    if remove_extra_whitespace:
        result = re.sub(r'\s+', ' ', result).strip()
    
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
    sentences = re.split(r'[.!?]+', text)
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
        ngram = ' '.join(tokens[i:i + n])
        ngrams.append(ngram)
    
    return ngrams


def contains_pattern(text: str, pattern: str, case_sensitive: bool = False) -> bool:
    """Check if text contains a pattern"""
    if not isinstance(text, str) or not isinstance(pattern, str):
        return False
    
    flags = 0 if case_sensitive else re.IGNORECASE
    return bool(re.search(pattern, text, flags))


def remove_stopwords(tokens: List[str], stopwords: Optional[List[str]] = None) -> List[str]:
    """Remove stopwords from token list"""
    if stopwords is None:
        # Basic English stopwords
        stopwords_set = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
        }
    else:
        stopwords_set = set(stopwords)
    
    return [token for token in tokens if token.lower() not in stopwords_set]


def compute_token_frequencies(frames, stop_words: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
    """
    Compute token frequencies across multiple DocDataFrame or DocLazyFrame objects.
    
    This function tokenizes the document column of each frame and calculates
    token frequencies within each frame. All frequency dictionaries share the
    same set of keys (tokens) for consistent comparison.
    
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
    Dict[str, Dict[str, int]]
        Dictionary mapping frame names to frequency dictionaries.
        Each frequency dictionary maps tokens to their frequency counts within that frame.
        All frequency dictionaries have the same set of keys (union of all tokens).
        
    Examples
    --------
    >>> import docframe as dp
    >>> df1 = dp.DocDataFrame({"text": ["hello world", "hello there"]})
    >>> df2 = dp.DocDataFrame({"text": ["world peace", "hello world"]})
    >>> frames = {"frame1": df1, "frame2": df2}
    >>> frequencies = dp.compute_token_frequencies(frames)
    >>> list(frequencies.keys())  # Frame names
    ['frame1', 'frame2']
    >>> sorted(frequencies['frame1'].keys())  # Same keys in both
    ['hello', 'peace', 'there', 'world']
    >>> frequencies['frame1']['hello']  # Count in first frame
    2
    >>> frequencies['frame2']['hello']  # Count in second frame  
    1
    
    >>> # With stop words
    >>> stop_words = ['hello']
    >>> frequencies = dp.compute_token_frequencies(frames, stop_words=stop_words)
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
    """
    if not frames:
        raise ValueError("At least one frame must be provided")
    
    # Import here to avoid circular imports
    from .docframe import DocDataFrame, DocLazyFrame
    
    # Validate input types
    for name, frame in frames.items():
        if not isinstance(frame, (DocDataFrame, DocLazyFrame)):
            raise TypeError(f"Frame '{name}' must be DocDataFrame or DocLazyFrame, got {type(frame)}")
    
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
        for tokens in doc_series.text.tokenize().to_list():
            if tokens:  # Skip empty token lists
                # Filter out stop words
                filtered_tokens = [token for token in tokens if token not in stop_words_set]
                tokens_list.extend(filtered_tokens)
                all_tokens.update(filtered_tokens)
        
        frame_tokens_lists[name] = tokens_list
    
    # Create frequency dictionaries with consistent keys
    result = {}
    for name, tokens_list in frame_tokens_lists.items():
        # Count tokens in this frame
        freq_dict = {}
        for token in tokens_list:
            freq_dict[token] = freq_dict.get(token, 0) + 1
        
        # Ensure all tokens are represented (with 0 for missing tokens)
        complete_freq_dict = {token: freq_dict.get(token, 0) for token in sorted(all_tokens)}
        result[name] = complete_freq_dict
    
    return result
