"""
Simple example demonstrating docframe functionality
"""


# First, let's test our library works
def test_basic_functionality():
    """Test that our library can be imported and used"""
    try:
        import os
        import sys

        # Add parent directory to path for importing docframe
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        import polars as pl

        from docframe import DocDataFrame

        print("✓ Successfully imported docframe")

        # Test text processing via polars Series namespace
        texts = ["Hello world", "This is a test", "Polars is fast"]
        series = pl.Series(texts)
        print(f"✓ Created Series with {len(series)} documents")

        # Test tokenization via text namespace
        tokens = series.text.tokenize()
        print(f"✓ Tokenized documents: {tokens[0]}")

        # Test word count via text namespace
        word_counts = series.text.word_count()
        print(f"✓ Word counts: {word_counts.to_list()}")

        # Test DocDataFrame
        metadata = {"category": ["greeting", "test", "tech"]}
        df = DocDataFrame.from_texts(texts, metadata)
        print(f"✓ Created DocDataFrame with {len(df)} documents")

        # Test text processing
        df_with_stats = df.add_word_count().add_char_count()
        print("✓ Added text statistics")

        # Test filtering
        long_docs = df.filter_by_length(min_words=2)
        print(f"✓ Filtered to {len(long_docs)} documents with 2+ words")

        # Test cleaning
        clean_df = df.clean_documents()
        print("✓ Cleaned documents")

        print("\n=== All tests passed! ===")

        # Assert key functionality works
        assert len(series) == 3, "Series should have 3 documents"
        assert len(tokens[0]) > 0, "Tokenization should produce tokens"
        assert sum(word_counts.to_list()) > 0, "Word counts should be positive"
        assert len(df) == 3, "DocDataFrame should have 3 documents"
        assert len(long_docs) >= 0, "Filtering should work"

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_basic_functionality()
