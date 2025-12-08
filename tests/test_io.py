"""Test I/O operations for DocFrame."""

import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import docframe
import docframe.utils as df_utils
from docframe import DocDataFrame


class TestIOOperations:
    """Test input/output operations"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pl.DataFrame({
            "article": [
                "The quick brown fox jumps over the lazy dog",
                "Pack my box with five dozen liquor jugs",
                "How vexingly quick daft zebras jump",
            ],
            "author": ["Alice", "Bob", "Charlie"],
            "year": [2020, 2021, 2022],
        })

    @pytest.fixture
    def temp_csv(self, sample_data):
        """Create a temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_data.write_csv(f.name)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_text_file(self):
        """Create a temporary text file"""
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            tmp_file.write("Plain text document for testing.")
            tmp_file.flush()
            file_path = tmp_file.name
        finally:
            tmp_file.close()

        try:
            yield file_path
        finally:
            os.unlink(file_path)

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

    def test_read_zip(self):
        """Read text files from a ZIP archive into a DocDataFrame."""

        zip_path = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "data"
            / "zip_example"
            / "data.zip"
        )

        doc_df = docframe.read_zip(zip_path)

        assert isinstance(doc_df, DocDataFrame)
        assert doc_df.active_document_name == "document"

        df = doc_df.dataframe
        # Archive contains three text files: 1.txt, 2.md, 3 (no extension)
        assert df.shape == (3, 4)
        assert set(df.columns) == {"file_path", "base_name", "extension", "document"}

        assert sorted(df["base_name"].to_list()) == ["1", "2", "3"]
        assert sorted(df["extension"].to_list()) == ["", ".md", ".txt"]
        sample_row = (
            df.filter(pl.col("base_name") == "1").select("document").to_series().item()
        )
        assert "Eldoria" in sample_row

        raw_df = docframe.read_zip(zip_path, document_column=False)
        assert isinstance(raw_df, pl.DataFrame)
        assert "document" in raw_df.columns

    def test_read_text_single_column(self, temp_text_file):
        """Plain text files should produce DocDataFrames with a single document column."""

        doc_df = docframe.read_text(temp_text_file)

        assert isinstance(doc_df, DocDataFrame)
        assert doc_df.active_document_name == "document"

        df = doc_df.dataframe
        assert df.shape == (1, 1)
        assert df.columns == ["document"]
        assert df["document"].item(0) == "Plain text document for testing."

    def test_read_text_without_docframe(self, temp_text_file):
        """document_column=False should return a plain Polars DataFrame."""

        raw_df = docframe.read_text(temp_text_file, document_column=False)

        assert isinstance(raw_df, pl.DataFrame)
        assert raw_df.shape == (1, 1)
        assert raw_df.columns == ["document"]
        assert raw_df["document"].item(0) == "Plain text document for testing."

    def test_excel_sheet_names_use_polars_default_engine(self, monkeypatch):
        """DocFrame should defer to Polars defaults without forcing an engine."""

        ensured = {"called": False}
        captured_kwargs: list[dict[str, Any]] = []

        def fake_ensure():
            ensured["called"] = True

        def fake_read_excel(*args, **kwargs):
            captured_kwargs.append(kwargs.copy())
            assert kwargs.get("sheet_id") is None
            return {
                "Alpha": pl.DataFrame({"x": [1]}),
                "Beta": pl.DataFrame({"x": [2]}),
            }

        monkeypatch.setattr(df_utils, "_ensure_fastexcel_available", fake_ensure)
        monkeypatch.setattr(df_utils, "_read_excel", fake_read_excel)

        sheets = docframe.excel_sheet_names("dummy.xlsx")

        assert ensured["called"] is True
        assert captured_kwargs == [{"sheet_id": None}]
        assert sheets == ["Alpha", "Beta"]

    def test_excel_sheet_names_single_sheet_dataframe(self, monkeypatch):
        """When Polars returns a DataFrame, fallback sheet naming should apply."""

        monkeypatch.setattr(df_utils, "_ensure_fastexcel_available", lambda: None)
        monkeypatch.setattr(
            df_utils,
            "_read_excel",
            lambda *args, **kwargs: pl.DataFrame({"x": [1, 2]}),
        )

        sheets = docframe.excel_sheet_names("solo.xlsx")

        assert sheets == ["Sheet1"]

    def test_read_excel_strips_engine_kwarg(self, monkeypatch, tmp_path):
        """DocFrame should remove any caller-provided engine hint."""

        ensured = {"called": False}
        captured_kwargs: list[dict[str, Any]] = []

        def fake_ensure():
            ensured["called"] = True

        def fake_read_excel(*args, **kwargs):
            captured_kwargs.append(kwargs.copy())
            return pl.DataFrame({"value": [1]})

        monkeypatch.setattr(df_utils, "_ensure_fastexcel_available", fake_ensure)
        monkeypatch.setattr(df_utils, "_read_excel", fake_read_excel)

        result = docframe.read_excel(
            tmp_path / "sample.xlsx",
            document_column=False,
            engine="openpyxl",
        )

        assert ensured["called"] is True
        assert captured_kwargs == [{}]
        assert isinstance(result, pl.DataFrame)

    def test_read_excel_missing_engine_dependency(self, monkeypatch, tmp_path):
        """Module import errors from Polars should be surfaced as ImportError."""

        monkeypatch.setattr(df_utils, "_ensure_fastexcel_available", lambda: None)

        def fake_read_excel(*_, **__):
            raise ModuleNotFoundError("fastexcel not installed")

        monkeypatch.setattr(df_utils, "_read_excel", fake_read_excel)

        with pytest.raises(ImportError) as exc:
            docframe.read_excel(tmp_path / "missing.xlsx", document_column=False)

        assert "Polars could not import its default Excel engine" in str(exc.value)
