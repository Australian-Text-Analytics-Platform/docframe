"""
Test the new polars text namespace functionality
"""

import os
import sys

import polars as pl

# Add the project path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import docframe to trigger namespace registration
import docframe


def test_expr_namespace():
    """Test that pl.col().text works"""
    df = pl.DataFrame({"text": ["Hello World!", "This is a test.", "Another example."]})

    # Test expression namespace
    result = df.select(
        pl.col("text").text.word_count().alias("words"),
        pl.col("text").text.char_count().alias("chars"),
        pl.col("text").text.tokenize().alias("tokens"),
    )

    print("Expression namespace test:")
    print(result)
    print()


def test_series_namespace():
    """Test that series.text works"""
    series = pl.Series("text", ["Hello World!", "This is a test.", "Another example."])

    # Test series namespace
    word_counts = series.text.word_count()
    tokens = series.text.tokenize()

    print("Series namespace test:")
    print("Word counts:", word_counts)
    print("Tokens:", tokens)
    print()


def test_dataframe_namespace():
    """Test that df.text works"""
    df = pl.DataFrame(
        {
            "text": ["Hello World!", "This is a test.", "Another example."],
            "id": [1, 2, 3],
        }
    )

    # Test dataframe namespace
    result = df.text.word_count("text")

    print("DataFrame namespace test:")
    print(result)
    print()


def test_document_shortcut():
    """Test df.document.text works with DocDataFrame"""
    from docframe import DocDataFrame

    df = DocDataFrame(
        {
            "text": ["Hello World!", "This is a test.", "Another example."],
            "id": [1, 2, 3],
        }
    )

    # Test document shortcut with text namespace
    word_counts = df.document.text.word_count()

    print("Document shortcut test:")
    print("Word counts:", word_counts)
    print()


def test_namespace_conversions():
    """Test namespace conversion methods"""
    from docframe import DocDataFrame

    # Test DataFrame.text.to_docdataframe()
    regular_df = pl.DataFrame(
        {
            "article": [
                "The quick brown fox",
                "Jumps over the lazy dog",
                "Pack my box with five dozen liquor jugs",
            ],
            "author": ["Alice", "Bob", "Charlie"],
            "year": [2020, 2021, 2022],
        }
    )

    doc_df = regular_df.text.to_docdataframe(document_column="article")
    print("DataFrame namespace conversion test:")
    print(
        f"Converted to DocDataFrame with document column: '{doc_df.active_document_name}'"
    )

    # Test auto-detection
    doc_df_auto = regular_df.text.to_docdataframe()
    print(f"Auto-detection picked: '{doc_df_auto.active_document_name}'")

    # Test Series text processing directly via namespace
    regular_series = pl.Series(
        "texts", ["First document", "Second document", "Third document"]
    )
    # Use text namespace directly on series for text processing
    word_counts = regular_series.text.word_count()
    print(f"Series text processing: word counts = {word_counts.to_list()}")
    print()


if __name__ == "__main__":
    print("Testing polars text namespace registration...")
    print()

    try:
        test_expr_namespace()
        test_series_namespace()
        test_dataframe_namespace()
        test_document_shortcut()
        test_namespace_conversions()

        print("All tests passed! ✅")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
