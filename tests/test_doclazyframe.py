"""
Test suite for DocLazyFrame functionality
"""

import polars as pl
import pytest

from docframe import DocDataFrame, DocLazyFrame


class TestDocLazyFrame:
    """Test DocLazyFrame functionality"""

    def test_init_from_lazyframe(self):
        """Test initialization from polars LazyFrame"""
        df = pl.DataFrame(
            {
                "document": ["Hello world", "This is a test"],
                "author": ["Alice", "Bob"],
            }
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="document")
        assert doc_lf.document_column == "document"
        assert doc_lf.active_document_name == "document"

    def test_guess_document_column(self):
        """Test automatic document column detection for LazyFrame"""
        data = {
            "id": [1, 2, 3],
            "title": ["Short", "Another", "Brief"],
            "content": [
                "This is a much longer document with detailed content",
                "Another very long piece of text with substantial content",
                "A third extensive document containing detailed information",
            ],
        }
        df = pl.DataFrame(data).lazy()

        # Should detect 'content' as document column
        doc_lf = DocLazyFrame(df)
        assert doc_lf.document_column == "content"

    def test_collect_to_docdataframe(self):
        """Test collecting LazyFrame to DocDataFrame"""
        df = pl.DataFrame(
            {
                "document": ["Hello world", "This is a test"],
                "author": ["Alice", "Bob"],
            }
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="document")
        doc_df = doc_lf.collect()

        assert isinstance(doc_df, DocDataFrame)
        assert doc_df.active_document_name == "document"
        assert len(doc_df) == 2

    def test_to_docdataframe(self):
        """Test converting to DocDataFrame"""
        df = pl.DataFrame(
            {
                "document": ["Hello world", "This is a test"],
                "author": ["Alice", "Bob"],
            }
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="document")
        doc_df = doc_lf.to_docdataframe()

        assert isinstance(doc_df, DocDataFrame)
        assert doc_df.active_document_name == "document"

    def test_to_lazyframe(self):
        """Test converting to regular polars LazyFrame"""
        df = pl.DataFrame(
            {
                "document": ["Hello world", "This is a test"],
                "author": ["Alice", "Bob"],
            }
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="document")
        lf = doc_lf.to_lazyframe()

        assert isinstance(lf, pl.LazyFrame)
        # Should preserve all columns - use collect_schema to avoid performance warning
        column_names = lf.collect_schema().names()
        assert "document" in column_names
        assert "author" in column_names

    def test_with_document_column(self):
        """Test changing document column"""
        df = pl.DataFrame(
            {
                "text1": ["Hello world", "This is a test"],
                "text2": ["Another text", "More content"],
                "author": ["Alice", "Bob"],
            }
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="text1")
        doc_lf2 = doc_lf.with_document_column("text2")

        assert doc_lf2.document_column == "text2"
        assert doc_lf.document_column == "text1"  # Original unchanged

    def test_document_property(self):
        """Test document property returns expression"""
        df = pl.DataFrame(
            {
                "document": ["Hello world", "This is a test"],
                "author": ["Alice", "Bob"],
            }
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="document")
        doc_expr = doc_lf.document

        assert isinstance(doc_expr, pl.Expr)

    def test_serialization(self):
        """Test LazyFrame serialization"""
        df = pl.DataFrame(
            {
                "document": ["Hello world", "This is a test"],
                "author": ["Alice", "Bob"],
            }
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="document")

        # Test JSON serialization
        json_str = doc_lf.serialize(format="json")
        assert isinstance(json_str, str)
        assert "document" in json_str

        # Test deserialization
        doc_lf2 = DocLazyFrame.deserialize(json_str, format="json")
        assert doc_lf2.document_column == "document"

    def test_polars_method_delegation(self):
        """Test that polars LazyFrame methods work"""
        df = pl.DataFrame(
            {
                "document": ["Hello world", "This is a test", "Another doc"],
                "author": ["Alice", "Bob", "Charlie"],
                "year": [2020, 2021, 2022],
            }
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="document")

        # Test filter
        filtered = doc_lf.filter(pl.col("year") > 2020)
        assert isinstance(filtered, DocLazyFrame)
        assert filtered.document_column == "document"

        # Test select
        selected = doc_lf.select(["document", "author"])
        assert isinstance(selected, DocLazyFrame)
        assert "year" not in selected.columns

        # Test with_columns
        with_cols = doc_lf.with_columns(pl.col("year").alias("publication_year"))
        assert isinstance(with_cols, DocLazyFrame)
        assert "publication_year" in with_cols.columns

    def test_repr_and_str(self):
        """Test string representations"""
        df = pl.DataFrame(
            {
                "document": ["Hello world", "This is a test"],
                "author": ["Alice", "Bob"],
            }
        ).lazy()

        doc_lf = DocLazyFrame(df, document_column="document")

        repr_str = repr(doc_lf)
        assert "DocLazyFrame" in repr_str
        assert "document_column='document'" in repr_str

        str_str = str(doc_lf)
        assert "Document column:" in str_str

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        df = pl.DataFrame(
            {
                "document": ["Hello world", "This is a test"],
                "author": ["Alice", "Bob"],
            }
        ).lazy()

        # Test invalid document column
        with pytest.raises(ValueError, match="Document column 'nonexistent' not found"):
            DocLazyFrame(df, document_column="nonexistent")

        # Test non-string document column
        df_numeric = pl.DataFrame(
            {
                "numbers": [1, 2, 3],
                "text": ["a", "b", "c"],
            }
        ).lazy()

        with pytest.raises(ValueError, match="is not a string column"):
            DocLazyFrame(df_numeric, document_column="numbers")
