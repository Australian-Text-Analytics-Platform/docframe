"""
Test suite for docframe
"""

import io

import polars as pl
import pytest

from docframe import DocDataFrame, DocLazyFrame


class TestDocDataFrame:
    """Test DocDataFrame functionality"""

    def test_init_from_dict(self):
        data = {
            "document": ["Hello world", "This is a test"],
            "author": ["Alice", "Bob"],
        }
        doc_df = DocDataFrame(data)
        assert len(doc_df) == 2
        assert doc_df.active_document_name == "document"

    def test_from_texts(self):
        texts = ["Hello world", "This is a test"]
        metadata = {"author": ["Alice", "Bob"]}
        doc_df = DocDataFrame.from_texts(texts, metadata)
        assert len(doc_df) == 2
        assert "author" in doc_df.dataframe.columns

    def test_document_property(self):
        data = {"document": ["Hello world", "This is a test"]}
        doc_df = DocDataFrame(data)
        doc_series = doc_df.document
        assert isinstance(doc_series, pl.Series)
        assert len(doc_series) == 2

    def test_add_word_count(self):
        data = {"document": ["Hello world", "This is a test document"]}
        doc_df = DocDataFrame(data)
        result = doc_df.add_word_count()
        assert "word_count" in result.dataframe.columns
        word_counts = result.dataframe["word_count"].to_list()
        assert word_counts == [2, 5]

    def test_filter_by_length(self):
        data = {
            "document": ["Hi", "Hello world", "This is a longer text"],
            "id": [1, 2, 3],
        }
        doc_df = DocDataFrame(data)
        filtered = doc_df.filter_by_length(min_words=2)
        assert len(filtered) == 2  # Should exclude "Hi"

    def test_filter_by_pattern(self):
        data = {
            "document": ["Hello world", "Goodbye moon", "Hello again"],
            "id": [1, 2, 3],
        }
        doc_df = DocDataFrame(data)
        filtered = doc_df.filter_by_pattern("Hello")
        assert len(filtered) == 2  # Should match first and third

    def test_clean_documents(self):
        data = {"document": ["Hello World!", "This is a TEST."]}
        doc_df = DocDataFrame(data)
        cleaned = doc_df.clean_documents()
        docs = cleaned.document.to_list()
        assert docs[0] == "hello world"
        assert docs[1] == "this is a test"

    def test_with_columns(self):
        data = {"document": ["Hello world", "This is a test"]}
        doc_df = DocDataFrame(data)
        # Use polars with_columns method via delegation
        result = doc_df.with_columns(pl.lit("greeting").alias("category"))
        assert "category" in result.dataframe.columns

    def test_sample(self):
        data = {
            "document": ["Text 1", "Text 2", "Text 3", "Text 4"],
            "id": [1, 2, 3, 4],
        }
        doc_df = DocDataFrame(data)
        sampled = doc_df.sample(n=2, seed=42)
        assert len(sampled) == 2

    def test_custom_document_column(self):
        data = {"text": ["Hello world", "This is a test"], "author": ["Alice", "Bob"]}
        doc_df = DocDataFrame(data, document_column="text")
        assert doc_df.active_document_name == "text"
        assert isinstance(doc_df.document, pl.Series)

    def test_serialize_json(self):
        """Test JSON serialization and deserialization"""

        data = {
            "document": ["Hello world", "This is a test document"],
            "author": ["Alice", "Bob"],
            "id": [1, 2],
        }
        doc_df = DocDataFrame(data, document_column="document")

        # Test JSON serialization
        json_data = doc_df.serialize(format="json")
        assert isinstance(json_data, str)
        assert len(json_data) > 0

        # Test JSON deserialization
        restored_df = DocDataFrame.deserialize(json_data, format="json")
        assert restored_df.active_document_name == "document"
        assert len(restored_df) == len(doc_df)
        assert restored_df.document.to_list() == doc_df.document.to_list()

        # Check that other columns are preserved
        assert "author" in restored_df.columns
        assert "id" in restored_df.columns
        assert restored_df.to_polars()["author"].to_list() == ["Alice", "Bob"]

    def test_serialize_json(self):
        """Test JSON serialization and deserialization"""

        data = {
            "document": ["Hello world", "This is a test document"],
            "author": ["Alice", "Bob"],
            "score": [0.8, 0.9],
        }
        doc_df = DocDataFrame(data, document_column="document")

        # Test JSON serialization
        json_data = doc_df.serialize(format="json")
        assert isinstance(json_data, str)
        assert len(json_data) > 0

        # Test JSON deserialization
        restored_df = DocDataFrame.deserialize(io.StringIO(json_data), format="json")
        assert restored_df.active_document_name == "document"
        assert len(restored_df) == len(doc_df)
        assert restored_df.document.to_list() == doc_df.document.to_list()

        # Check that other columns are preserved
        assert "author" in restored_df.columns
        assert "score" in restored_df.columns
        assert restored_df.to_polars()["author"].to_list() == ["Alice", "Bob"]
        assert restored_df.to_polars()["score"].to_list() == [0.8, 0.9]

    def test_serialize_to_file_json(self):
        """Test serialization to JSON file"""
        import os
        import tempfile

        data = {
            "document": ["Hello world", "This is a test"],
            "category": ["greeting", "test"],
        }
        doc_df = DocDataFrame(data, document_column="document")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Serialize to file
            result = doc_df.serialize(tmp_path, format="json")
            assert result is None  # Should return None when writing to file
            assert os.path.exists(tmp_path)

            # Deserialize from file
            restored_df = DocDataFrame.deserialize(tmp_path, format="json")
            assert restored_df.active_document_name == "document"
            assert len(restored_df) == len(doc_df)
            assert restored_df.document.to_list() == doc_df.document.to_list()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_doclazyframe_serialization(self):
        """Test DocLazyFrame serialization"""
        from docframe.core.docframe import DocLazyFrame

        # Create a DocLazyFrame
        df = pl.DataFrame(
            {"document": ["Hello world", "This is a test"], "id": [1, 2]}
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="document")
        assert doc_lf.active_document_name == "document"

        # Test JSON serialization with DocLazyFrame
        json_data = doc_lf.serialize(format="json")
        assert isinstance(json_data, str)

        # Test deserialization
        restored_lf = DocLazyFrame.deserialize(json_data, format="json")
        assert restored_lf.active_document_name == "document"

        # Collect and compare
        original_collected = doc_lf.collect()
        restored_collected = restored_lf.collect()
        assert (
            original_collected.document.to_list()
            == restored_collected.document.to_list()
        )

    def test_dataframe_lazyframe_conversion(self):
        """Test conversion between DocDataFrame and DocLazyFrame"""
        # Create test data
        data = {"text": ["hello world", "test document"], "id": [1, 2]}
        df = DocDataFrame(data, document_column="text")

        # Test to_doclazyframe
        lazy_df = df.to_doclazyframe()
        assert isinstance(lazy_df, DocLazyFrame)

        # Test to_docdataframe
        eager_df = lazy_df.to_docdataframe()
        assert isinstance(eager_df, DocDataFrame)

        # Test roundtrip data consistency
        assert df._df.equals(eager_df._df)

    def test_doclazyframe_consistency(self):
        """Test that DocDataFrame and DocLazyFrame have consistent APIs"""
        # Test data
        data = {
            "short": ["a", "b"],
            "long_text": ["this is much longer", "another long text"],
        }
        df = DocDataFrame(data)
        lazy_df = DocLazyFrame(pl.DataFrame(data).lazy())

        # Both should have the same properties
        assert df.document_column == lazy_df.document_column
        assert df.active_document_name == lazy_df.active_document_name
        assert df.columns == lazy_df.columns

        # Both should have guess_document_column classmethod
        df_guess = DocDataFrame.guess_document_column(pl.DataFrame(data))
        lazy_guess = DocLazyFrame.guess_document_column(pl.DataFrame(data))
        assert df_guess == lazy_guess

        # DocLazyFrame should have to_lazyframe method
        assert isinstance(lazy_df.to_lazyframe(), pl.LazyFrame)

        # Both should have proper string representations
        assert "Document column:" in str(df)
        assert "Document column:" in str(lazy_df)

    def test_serialize_custom_document_column(self):
        """Test serialization preserves custom document column name"""

        data = {
            "text_content": ["Hello world", "This is a test"],
            "author": ["Alice", "Bob"],
        }
        doc_df = DocDataFrame(data, document_column="text_content")

        # Test JSON serialization
        json_data = doc_df.serialize(format="json")
        assert isinstance(json_data, str)
        restored_json_df = DocDataFrame.deserialize(json_data, format="json")
        assert restored_json_df.active_document_name == "text_content"

    def test_serialize_invalid_format(self):
        """Test serialization with invalid format raises error"""
        data = {"document": ["Hello world"]}
        doc_df = DocDataFrame(data)

        with pytest.raises(ValueError, match="Unsupported format"):
            doc_df.serialize(format="invalid")

        with pytest.raises(ValueError, match="Unsupported format"):
            DocDataFrame.deserialize(io.BytesIO(b"test"), format="invalid")

    def test_concordance_functionality(self):
        """Test concordance functionality in text namespace"""
        data = {
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "A quick brown fox is very fast",
                "The fox runs quickly through the forest",
            ],
            "id": [1, 2, 3],
        }
        df = pl.DataFrame(data)

        # Test basic concordance
        concordance_result = df.text.concordance("text", "fox")

        assert isinstance(concordance_result, pl.DataFrame)
        assert len(concordance_result) == 3  # Should find 3 matches
        assert "document_idx" in concordance_result.columns
        assert "left_context" in concordance_result.columns
        assert "matched_text" in concordance_result.columns
        assert "right_context" in concordance_result.columns

        # Check that all matched_text are "fox"
        matched_texts = concordance_result["matched_text"].to_list()
        assert all(text == "fox" for text in matched_texts)

        # Test case sensitive search
        concordance_case = df.text.concordance("text", "Fox", case_sensitive=True)
        assert (
            len(concordance_case) == 0
        )  # Should find no matches with case sensitivity

        # Test regex search
        concordance_regex = df.text.concordance("text", r"fox|dog", regex=True)
        assert len(concordance_regex) == 4  # Should find both "fox" and "dog"

        # Test with limited context
        concordance_limited = df.text.concordance(
            "text", "fox", num_left_tokens=2, num_right_tokens=2
        )
        assert len(concordance_limited) == 3

        # Test empty search word
        concordance_empty = df.text.concordance("text", "")
        assert len(concordance_empty) == 0
        assert list(concordance_empty.columns) == [
            "document_idx",
            "left_context",
            "matched_text",
            "right_context",
            "l1",
            "l1_freq",
            "r1",
            "r1_freq",
        ]

    def test_frequency_analysis_basic(self):
        """Test basic frequency analysis functionality"""
        # Create test data with dates
        dates = [
            "2023-01-15",
            "2023-01-20",
            "2023-02-10",
            "2023-02-15",
            "2023-03-05",
            "2023-03-20",
        ]
        data = {
            "document": ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
            "created_at": dates,
            "category": ["A", "B", "A", "B", "A", "B"],
        }

        df = pl.DataFrame(data).with_columns(
            pl.col("created_at").str.to_datetime("%Y-%m-%d")
        )

        # Test monthly frequency using text namespace
        monthly_freq = df.text.frequency_analysis("created_at", frequency="monthly")
        assert "frequency_count" in monthly_freq.columns
        assert "time_period" in monthly_freq.columns
        assert "time_period_formatted" in monthly_freq.columns
        assert len(monthly_freq) == 3  # Jan, Feb, Mar

        # Check frequency counts
        monthly_data = monthly_freq.sort("time_period")
        counts = monthly_data.get_column("frequency_count").to_list()
        assert counts == [2, 2, 2]  # 2 docs each month

    def test_frequency_analysis_with_grouping(self):
        """Test frequency analysis with grouping columns"""
        dates = [
            "2023-01-15",
            "2023-01-20",
            "2023-02-10",
            "2023-02-15",
            "2023-03-05",
            "2023-03-20",
        ]
        data = {
            "document": ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
            "created_at": dates,
            "category": ["A", "B", "A", "B", "A", "B"],
        }

        df = pl.DataFrame(data).with_columns(
            pl.col("created_at").str.to_datetime("%Y-%m-%d")
        )

        # Test with grouping by category using text namespace
        grouped_freq = df.text.frequency_analysis(
            "created_at", group_by_columns=["category"], frequency="monthly"
        )

        assert len(grouped_freq) == 6  # 3 months × 2 categories

        # Check that we have both categories for each month
        grouped_data = grouped_freq.sort(["time_period", "category"])
        categories = grouped_data.get_column("category").to_list()
        assert categories == ["A", "B", "A", "B", "A", "B"]

    def test_frequency_analysis_different_frequencies(self):
        """Test different frequency options"""
        # Create data spanning multiple days, weeks, months
        dates = [
            "2023-01-01",
            "2023-01-02",
            "2023-01-08",
            "2023-01-15",
            "2023-02-01",
            "2023-02-15",
            "2023-03-01",
        ]
        data = {"document": [f"doc{i}" for i in range(1, 8)], "created_at": dates}

        df = pl.DataFrame(data).with_columns(
            pl.col("created_at").str.to_datetime("%Y-%m-%d")
        )

        # Test daily frequency using text namespace
        daily_freq = df.text.frequency_analysis("created_at", frequency="daily")
        assert len(daily_freq) == 7  # 7 unique days

        # Test weekly frequency
        weekly_freq = df.text.frequency_analysis("created_at", frequency="weekly")
        assert len(weekly_freq) <= 7  # Should be fewer weeks than days

        # Test monthly frequency
        monthly_freq = df.text.frequency_analysis("created_at", frequency="monthly")
        assert len(monthly_freq) == 3  # Jan, Feb, Mar

        # Test yearly frequency
        yearly_freq = df.text.frequency_analysis("created_at", frequency="yearly")
        assert len(yearly_freq) == 1  # Only 2023

    def test_frequency_analysis_sorting(self):
        """Test sorting options in frequency analysis"""
        dates = ["2023-03-01", "2023-01-01", "2023-02-01"]
        data = {"document": ["doc1", "doc2", "doc3"], "created_at": dates}

        df = pl.DataFrame(data).with_columns(
            pl.col("created_at").str.to_datetime("%Y-%m-%d")
        )

        # Test with sorting (default) using text namespace
        sorted_freq = df.text.frequency_analysis("created_at", frequency="monthly")
        periods = sorted_freq.get_column("time_period_formatted").to_list()
        assert periods == ["2023-01", "2023-02", "2023-03"]

        # Test without sorting
        unsorted_freq = df.text.frequency_analysis(
            "created_at", frequency="monthly", sort_by_time=False
        )
        # Should have same data but potentially different order
        assert len(unsorted_freq) == 3

    def test_frequency_analysis_invalid_frequency(self):
        """Test error handling for invalid frequency"""
        data = {
            "document": ["doc1", "doc2"],
            "created_at": ["2023-01-01", "2023-01-02"],
        }

        df = pl.DataFrame(data).with_columns(
            pl.col("created_at").str.to_datetime("%Y-%m-%d")
        )

        with pytest.raises(ValueError, match="Unsupported frequency"):
            df.text.frequency_analysis("created_at", frequency="hourly")

    # ...existing code...


if __name__ == "__main__":
    pytest.main([__file__])
