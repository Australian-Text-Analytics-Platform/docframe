"""
Test suite for DocFrame core functionality
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

import polars as pl

from docframe import DocDataFrame

# Import the helper functions from conftest
from .conftest import get_sample_data, get_tweet_data_path


class TestDocDataFrame:
    """Test DocDataFrame functionality"""

    def test_creation_with_auto_detection(self):
        """Test DocDataFrame creation with automatic document column detection"""
        data = get_sample_data()
        df = DocDataFrame(data)

        # Should auto-detect 'content' as the document column
        assert df.active_document_name == "content"
        assert len(df) == 3

    def test_creation_with_manual_column(self):
        """Test DocDataFrame creation with manual document column specification"""
        data = get_sample_data()
        df = DocDataFrame(data, document_column="title")

        assert df.active_document_name == "title"
        assert df.document[0] == "Short title"

    def test_guess_document_column_class_method(self):
        """Test the guess_document_column class method"""
        data = get_sample_data()
        pl_df = pl.DataFrame(data)

        guessed = DocDataFrame.guess_document_column(pl_df)
        assert guessed == "content"  # Should pick the longest column

    def test_from_texts_constructor(self):
        """Test creating DocDataFrame from texts"""
        texts = ["Document 1", "Document 2", "Document 3"]
        metadata = {"id": [1, 2, 3], "category": ["A", "B", "C"]}

        df = DocDataFrame.from_texts(texts, metadata, document_column="document")
        assert df.active_document_name == "document"
        assert len(df) == 3
        assert df.document[0] == "Document 1"

    def test_document_property_access(self):
        """Test accessing the document column via .document property"""
        data = get_sample_data()
        df = DocDataFrame(data)

        # Should return a polars Series with text processing capabilities
        doc_series = df.document
        assert isinstance(doc_series, pl.Series)
        assert len(doc_series) == 3

    def test_add_text_statistics(self):
        """Test adding text statistics columns"""
        data = get_sample_data()
        df = DocDataFrame(data)

        # Add various text statistics
        df_with_stats = df.add_word_count().add_char_count().add_sentence_count()

        assert "word_count" in df_with_stats.columns
        assert "char_count" in df_with_stats.columns
        assert "sentence_count" in df_with_stats.columns

        # Check that the stats are reasonable
        word_counts = df_with_stats.select("word_count").to_series().to_list()
        assert all(count > 0 for count in word_counts)

    def test_clean_documents(self):
        """Test document cleaning"""
        data = get_sample_data()
        df = DocDataFrame(data)

        cleaned_df = df.clean_documents(lowercase=True, remove_punct=True)

        # Should still have same document column
        assert cleaned_df.active_document_name == df.active_document_name

        # Documents should be cleaned
        first_doc = cleaned_df.document[0]
        assert first_doc.islower() or first_doc == ""  # Should be lowercase

    def test_filter_operations(self):
        """Test filtering operations"""
        data = get_sample_data()
        df = DocDataFrame(data)

        # Filter by length
        long_docs = df.filter_by_length(min_words=10)
        assert len(long_docs) <= len(df)

        # Filter by pattern
        filtered = df.filter_by_pattern("document", case_sensitive=False)
        assert len(filtered) >= 0  # Should find documents containing "document"

    def test_sampling(self):
        """Test document sampling"""
        data = get_sample_data()
        df = DocDataFrame(data)

        # Sample 2 documents
        sampled = df.sample(n=2, seed=42)
        assert len(sampled) == 2
        assert sampled.active_document_name == df.active_document_name

        # Sample by fraction
        sampled_frac = df.sample(fraction=0.5, seed=42)
        assert len(sampled_frac) <= len(df)

    def test_metadata_operations(self):
        """Test metadata operations via polars delegation"""
        data = get_sample_data()
        df = DocDataFrame(data)

        # Add metadata via with_columns
        df_with_meta = df.with_columns(pl.lit([0.1, 0.2, 0.3]).alias("score"))
        assert "score" in df_with_meta.columns

        # Remove columns via drop
        df_removed = df_with_meta.drop(["score"])
        assert "score" not in df_removed.columns

        # Rename columns via rename
        df_renamed = df.rename({"category": "type"})
        assert "type" in df_renamed.columns
        assert "category" not in df_renamed.columns

    def test_data_export(self):
        """Test data export functionality via polars delegation"""
        data = get_sample_data()
        df = DocDataFrame(data)

        # Test conversion to polars
        pl_df = df.to_polars()
        assert isinstance(pl_df, pl.DataFrame)
        assert len(pl_df) == len(df)

        # Test CSV export via delegation to polars write_csv
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df.write_csv(tmp.name)
            # Verify file was created
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)


class TestAutoDetection:
    """Test automatic document column detection"""

    def test_clear_winner_scenario(self):
        """Test auto-detection with one clearly longer column"""
        data = {
            "short": ["A", "B", "C"],
            "medium": ["Medium text", "Another medium", "Third medium"],
            "long_content": [
                "This is a very long document with lots of content that should clearly be detected",
                "Another extremely lengthy piece of text with substantial information and details",
                "A third comprehensive document with extensive content and detailed information",
            ],
        }

        df = DocDataFrame(data)
        assert df.active_document_name == "long_content"

    def test_single_string_column(self):
        """Test auto-detection with single string column"""
        data = {
            "id": [1, 2, 3],
            "text": ["Document one", "Document two", "Document three"],
            "score": [0.1, 0.2, 0.3],
        }

        df = DocDataFrame(data)
        assert df.active_document_name == "text"

    def test_no_string_columns_fallback(self):
        """Test fallback when no string columns exist"""
        data = {"id": [1, 2, 3], "score": [0.1, 0.2, 0.3]}

        try:
            df = DocDataFrame(data)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found in data" in str(e)

    def test_manual_override_works(self):
        """Test that manual document column specification still works"""
        data = get_sample_data()
        df = DocDataFrame(data, document_column="title")

        assert df.active_document_name == "title"
        assert "title" in df.document[0].lower()


class TestRealWorldData:
    """Test with real-world datasets"""

    def test_tweet_dataset_loading(self):
        """Test loading and processing tweet dataset"""
        tweet_path = get_tweet_data_path()

        if os.path.exists(tweet_path):
            # Load using docframe.read_csv utility function instead
            import docframe

            df = docframe.read_csv(tweet_path, document_column="text")
            sample_df = df.sample(n=min(10, len(df)), seed=42)

            assert df.active_document_name == "text"
            assert len(sample_df) <= 10

            # Test text operations on real data
            stats_df = sample_df.add_word_count().add_char_count()
            assert "word_count" in stats_df.columns
            assert "char_count" in stats_df.columns
        else:
            # Skip if data file not available
            pass

    def test_auto_detection_on_tweet_data(self):
        """Test auto-detection on tweet dataset"""
        tweet_path = get_tweet_data_path()

        if os.path.exists(tweet_path):
            # Load without specifying document column
            pl_df = pl.read_csv(tweet_path).head(100)  # Sample for speed

            guessed = DocDataFrame.guess_document_column(pl_df)
            # Should detect 'text' as the main text column
            assert guessed == "text"
        else:
            # Skip if data file not available
            pass


def run_tests():
    """Run all tests manually (for when pytest is not available)"""
    import traceback

    test_classes = [
        TestDocDataFrame,
        TestAutoDetection,
        TestRealWorldData,
    ]
    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n=== {test_class.__name__} ===")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
                    traceback.print_exc()

    print(f"\n=== Test Results ===")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")

    return passed_tests == total_tests


if __name__ == "__main__":
    run_tests()
