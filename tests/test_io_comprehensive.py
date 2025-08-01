"""
Test suite for I/O operations and data conversion functionality
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from docframe import DocDataFrame, from_pandas, read_csv, read_json, read_parquet


class TestIOOperations:
    """Test comprehensive I/O operations"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return {
            "document": [
                "This is the first document with substantial content for testing",
                "Second document contains different text for analysis",
                "Third document has more varied content and examples",
            ],
            "category": ["news", "blog", "article"],
            "year": [2020, 2021, 2022],
            "author": ["Alice", "Bob", "Charlie"],
        }

    def test_read_csv_comprehensive(self, sample_data):
        """Test comprehensive CSV reading functionality"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create CSV content
            header = ",".join(sample_data.keys())
            f.write(header + "\n")

            for i in range(len(sample_data["document"])):
                row = [str(sample_data[col][i]) for col in sample_data.keys()]
                f.write(",".join(row) + "\n")

            temp_path = f.name

        try:
            # Test with explicit document column
            df = read_csv(temp_path, document_column="document")
            assert isinstance(df, DocDataFrame)
            assert df.active_document_name == "document"
            assert len(df) == 3

            # Test with auto-detection
            df_auto = read_csv(temp_path)
            assert isinstance(df_auto, DocDataFrame)
            assert df_auto.active_document_name == "document"  # Should auto-detect

            # Test with no document column
            df_none = read_csv(temp_path, document_column=False)
            assert isinstance(df_none, pl.DataFrame)  # Should return regular DataFrame

        finally:
            Path(temp_path).unlink()

    def test_read_parquet(self, sample_data):
        """Test Parquet reading functionality"""
        # Create a regular polars DataFrame first
        df = pl.DataFrame(sample_data)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            # Write parquet file
            df.write_parquet(temp_path)

            # Read with docframe
            doc_df = read_parquet(temp_path, document_column="document")
            assert isinstance(doc_df, DocDataFrame)
            assert doc_df.active_document_name == "document"
            assert len(doc_df) == 3

            # Test auto-detection
            doc_df_auto = read_parquet(temp_path)
            assert isinstance(doc_df_auto, DocDataFrame)
            assert doc_df_auto.active_document_name == "document"

        finally:
            Path(temp_path).unlink()

    def test_read_json(self, sample_data):
        """Test JSON reading functionality"""
        # Create a regular polars DataFrame first
        df = pl.DataFrame(sample_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Write JSON file
            df.write_json(temp_path)

            # Read with docframe
            doc_df = read_json(temp_path, document_column="document")
            assert isinstance(doc_df, DocDataFrame)
            assert doc_df.active_document_name == "document"
            assert len(doc_df) == 3

        finally:
            Path(temp_path).unlink()

    def test_from_pandas_conversion(self, sample_data):
        """Test conversion from pandas DataFrame"""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        # Create pandas DataFrame
        pdf = pd.DataFrame(sample_data)

        # Convert to DocDataFrame
        doc_df = from_pandas(pdf, document_column="document")
        assert isinstance(doc_df, DocDataFrame)
        assert doc_df.active_document_name == "document"
        assert len(doc_df) == 3

        # Test auto-detection
        doc_df_auto = from_pandas(pdf)
        assert isinstance(doc_df_auto, DocDataFrame)
        assert doc_df_auto.active_document_name == "document"

    def test_write_operations(self, sample_data):
        """Test writing DocDataFrame to various formats"""
        df = DocDataFrame(sample_data)

        # Test CSV writing
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            df.write_csv(csv_path)
            assert Path(csv_path).exists()

            # Read back and verify
            df_read = read_csv(csv_path, document_column="document")
            assert df_read.active_document_name == "document"
            assert len(df_read) == len(df)

        finally:
            Path(csv_path).unlink()

        # Test Parquet writing
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            df.write_parquet(parquet_path)
            assert Path(parquet_path).exists()

            # Read back and verify
            df_read = read_parquet(parquet_path, document_column="document")
            assert df_read.active_document_name == "document"
            assert len(df_read) == len(df)

        finally:
            Path(parquet_path).unlink()

    def test_scan_operations(self, sample_data):
        """Test lazy scan operations"""
        # Create CSV file
        df = pl.DataFrame(sample_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            df.write_csv(temp_path)

            # Test scan_csv
            from docframe import scan_csv

            lazy_df = scan_csv(temp_path, document_column="document")

            # Should return DocLazyFrame
            from docframe.core.docframe import DocLazyFrame

            assert isinstance(lazy_df, DocLazyFrame)
            assert lazy_df.document_column == "document"

            # Test collection
            collected = lazy_df.collect()
            assert isinstance(collected, DocDataFrame)
            assert collected.active_document_name == "document"

        finally:
            Path(temp_path).unlink()


class TestDataConversions:
    """Test data conversion functionality"""

    def test_to_polars_conversion(self):
        """Test conversion to regular polars DataFrame"""
        df = DocDataFrame(
            {"document": ["Hello world", "Test text"], "category": ["A", "B"]}
        )

        polars_df = df.to_polars()
        assert isinstance(polars_df, pl.DataFrame)
        assert "document" in polars_df.columns
        assert "category" in polars_df.columns
        assert len(polars_df) == 2

    def test_to_doclazyframe_conversion(self):
        """Test conversion to DocLazyFrame"""
        df = DocDataFrame(
            {"document": ["Hello world", "Test text"], "category": ["A", "B"]}
        )

        lazy_df = df.to_doclazyframe()
        from docframe.core.docframe import DocLazyFrame

        assert isinstance(lazy_df, DocLazyFrame)
        assert lazy_df.document_column == "document"

    def test_concat_documents_function(self):
        """Test concatenating DocDataFrames"""
        df1 = DocDataFrame(
            {"document": ["Hello world", "Test text"], "category": ["A", "B"]}
        )

        df2 = DocDataFrame(
            {"document": ["More text", "Final doc"], "category": ["C", "D"]}
        )

        from docframe import concat_documents

        concatenated = concat_documents([df1, df2])

        assert isinstance(concatenated, DocDataFrame)
        assert concatenated.active_document_name == "document"
        assert len(concatenated) == 4
        assert "category" in concatenated.columns

    def test_serialization_formats(self):
        """Test different serialization formats"""
        df = DocDataFrame(
            {
                "document": ["Hello world", "Test text"],
                "category": ["A", "B"],
                "year": [2020, 2021],
            }
        )

        # Test JSON serialization
        json_str = df.serialize(format="json")
        assert isinstance(json_str, str)
        assert "document" in json_str

        # Test deserialization
        df_restored = DocDataFrame.deserialize(json_str, format="json")
        assert isinstance(df_restored, DocDataFrame)
        assert df_restored.active_document_name == "document"
        assert len(df_restored) == len(df)
        assert df_restored.columns == df.columns


class TestErrorHandling:
    """Test error handling for I/O operations"""

    def test_invalid_file_paths(self):
        """Test handling of invalid file paths"""
        with pytest.raises(Exception):  # Should raise some kind of file error
            read_csv("nonexistent_file.csv")

    def test_invalid_document_column(self):
        """Test handling of invalid document column specification"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\nvalue1,value2\nvalue3,value4\n")
            temp_path = f.name

        try:
            with pytest.warns(
                UserWarning, match="Could not create DocDataFrame/DocLazyFrame"
            ):
                result = read_csv(temp_path, document_column="nonexistent")
                # Should return regular DataFrame when DocDataFrame creation fails
                assert isinstance(result, pl.DataFrame)
        finally:
            Path(temp_path).unlink()

    def test_invalid_serialization_format(self):
        """Test handling of invalid serialization format"""
        df = DocDataFrame({"document": ["Hello world", "Test text"]})

        with pytest.raises(ValueError, match="Unsupported format"):
            df.serialize(format="invalid_format")

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        # Creating empty DocDataFrame should raise error
        with pytest.raises(
            ValueError, match="Document column 'document' not found in data"
        ):
            DocDataFrame({})
