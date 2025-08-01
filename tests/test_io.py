"""
Test I/O operations for DocFrame
"""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

import docframe
from docframe import DocDataFrame


class TestIOOperations:
    """Test input/output operations"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pl.DataFrame(
            {
                "article": [
                    "The quick brown fox jumps over the lazy dog",
                    "Pack my box with five dozen liquor jugs",
                    "How vexingly quick daft zebras jump",
                ],
                "author": ["Alice", "Bob", "Charlie"],
                "year": [2020, 2021, 2022],
            }
        )

    @pytest.fixture
    def temp_csv(self, sample_data):
        """Create a temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_data.write_csv(f.name)
            yield f.name
        os.unlink(f.name)

    def test_read_csv_with_document_column(self, temp_csv):
        """Test reading CSV with specified document column"""
        doc_df = docframe.read_csv(temp_csv, document_column="article")

        assert isinstance(doc_df, DocDataFrame)
        assert doc_df.active_document_name == "article"
        assert len(doc_df) == 3

    def test_read_csv_without_document_column(self, temp_csv):
        """Test reading CSV with document_column=False returns regular DataFrame"""
        df = docframe.read_csv(temp_csv, document_column=False)

        assert isinstance(df, pl.DataFrame)
        assert not isinstance(df, DocDataFrame)

    def test_read_csv_with_auto_detection(self, temp_csv):
        """Test reading CSV with explicit auto-detection"""
        doc_df = docframe.read_csv(temp_csv, document_column=None)

        assert isinstance(doc_df, DocDataFrame)
        # Should detect 'article' as it has the longest average text length
        assert doc_df.active_document_name == "article"

    def test_read_csv_default_auto_detection(self, temp_csv):
        """Test reading CSV with default auto-detection (no document_column parameter)"""
        doc_df = docframe.read_csv(temp_csv)

        assert isinstance(doc_df, DocDataFrame)
        # Should detect 'article' as it has the longest average text length
        assert doc_df.active_document_name == "article"

    def test_from_pandas(self):
        """Test conversion from pandas DataFrame and Series"""
        pd = pytest.importorskip("pandas")

        # Test DataFrame conversion
        pandas_df = pd.DataFrame({"text": ["doc1", "doc2", "doc3"], "id": [1, 2, 3]})

        doc_df = docframe.from_pandas(pandas_df, document_column="text")
        assert isinstance(doc_df, DocDataFrame)
        assert doc_df.active_document_name == "text"
        assert len(doc_df) == 3

        # Test Series conversion
        pandas_series = pd.Series(["text1", "text2", "text3"], name="documents")
        doc_series = docframe.from_pandas(pandas_series, document_column="documents")
        assert isinstance(doc_series, pl.Series)
        assert len(doc_series) == 3

    def test_concat_documents(self):
        """Test concatenating multiple DocDataFrames"""
        df1 = DocDataFrame(
            {"text": ["doc1", "doc2"], "id": [1, 2]}, document_column="text"
        )

        df2 = DocDataFrame(
            {"text": ["doc3", "doc4"], "id": [3, 4]}, document_column="text"
        )

        concatenated = docframe.concat_documents([df1, df2])

        assert isinstance(concatenated, DocDataFrame)
        assert len(concatenated) == 4
        assert concatenated.active_document_name == "text"
        assert concatenated["id"].to_list() == [1, 2, 3, 4]

    def test_write_operations(self, sample_data):
        """Test write operations through polars delegation"""
        doc_df = DocDataFrame(sample_data, document_column="article")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV write via delegation to polars
            csv_path = Path(tmpdir) / "test.csv"
            doc_df.write_csv(str(csv_path))
            assert csv_path.exists()

            # Test Parquet write via delegation to polars
            parquet_path = Path(tmpdir) / "test.parquet"
            doc_df.write_parquet(str(parquet_path))
            assert parquet_path.exists()

            # Test JSON write via delegation to polars
            json_path = Path(tmpdir) / "test.json"
            doc_df.write_json(str(json_path))
            assert json_path.exists()

    def test_scan_operations(self, temp_csv):
        """Test lazy scan operations"""
        # Test scan_csv - should return DocLazyFrame for lazy operations
        from docframe.core.docframe import DocLazyFrame

        doc_lf = docframe.scan_csv(temp_csv)
        # scan_csv should return DocLazyFrame for lazy operations
        assert isinstance(doc_lf, DocLazyFrame)
        assert doc_lf.active_document_name == "article"  # Should auto-detect

        # Test disabling conversion to get raw LazyFrame
        lazy_df = docframe.scan_csv(temp_csv, document_column=False)
        assert isinstance(lazy_df, pl.LazyFrame)

        # Collect and verify
        df = lazy_df.collect()
        assert len(df) == 3
        assert "article" in df.columns
