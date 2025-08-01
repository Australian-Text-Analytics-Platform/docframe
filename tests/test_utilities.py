"""
Test the new utilities and document column management
"""

import os
import tempfile
import warnings

import pytest


def test_document_column_management():
    """Test document column management methods"""
    import os
    import sys

    # Add parent directory to path for importing docframe
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import polars as pl

    from docframe import DocDataFrame

    # Create test data
    data = {
        "text": ["hello world", "this is a test"],
        "content": ["longer content here for testing", "another piece of content"],
        "id": [1, 2],
    }

    df = DocDataFrame(data)  # Should auto-detect 'content'

    # Test active_document_name
    assert df.active_document_name == "content"

    # Test set_document
    df_new = df.set_document("text")
    assert df_new.active_document_name == "text"
    assert df.active_document_name == "content"  # Original unchanged

    # Test set_document with invalid column
    with pytest.raises(ValueError, match="Document column 'nonexistent' not found"):
        df.set_document("nonexistent")

    # Test set_document with non-string column
    with pytest.raises(ValueError, match="not a string column"):
        df.set_document("id")

    # Test rename_document
    df_renamed = df.rename_document("main_text")
    assert df_renamed.active_document_name == "main_text"
    assert "main_text" in df_renamed.columns
    assert "content" not in df_renamed.columns

    # Test rename_document with existing column name
    with pytest.raises(ValueError, match="already exists"):
        df.rename_document("text")

    print("✅ Document column management tests passed")


def test_read_csv_utility():
    """Test the read_csv utility function"""
    import os
    import sys

    # Add parent directory to path for importing docframe
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import polars as pl

    import docframe

    # Create a temporary CSV file
    test_data = {
        "id": [1, 2, 3],
        "title": ["Short title", "Another title", "Third title"],
        "content": [
            "This is a much longer piece of content for testing",
            "Another long content piece with more words",
            "Yet another lengthy content for comprehensive testing",
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df_temp = pl.DataFrame(test_data)
        df_temp.write_csv(f.name)
        csv_path = f.name

    try:
        # Test default auto-detection (no document_column parameter)
        df1 = docframe.read_csv(csv_path)
        assert hasattr(df1, "active_document_name")
        assert df1.active_document_name == "content"  # Should auto-detect longest

        # Test DocDataFrame with specified column
        df2 = docframe.read_csv(csv_path, document_column="content")
        assert hasattr(df2, "active_document_name")
        assert df2.active_document_name == "content"

        # Test DocDataFrame with explicit auto-detection
        df3 = docframe.read_csv(csv_path, document_column=None)
        assert hasattr(df3, "active_document_name")
        assert df3.active_document_name == "content"  # Should auto-detect longest

        # Test regular DataFrame (document_column=False)
        df3b = docframe.read_csv(csv_path, document_column=False)
        assert isinstance(df3b, pl.DataFrame)
        assert not hasattr(df3b, "active_document_name")

        # Test with non-existent column (should warn and return DataFrame)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df4 = docframe.read_csv(csv_path, document_column="nonexistent")
            assert len(w) == 1
            assert "not found" in str(w[-1].message)
            assert isinstance(df4, pl.DataFrame)
            assert not hasattr(df4, "active_document_name")

        # Test with non-string column (should warn and return DataFrame)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df5 = docframe.read_csv(csv_path, document_column="id")
            assert len(w) == 1
            assert "not a string column" in str(w[-1].message)
            assert isinstance(df5, pl.DataFrame)

        print("✅ read_csv utility tests passed")

    finally:
        # Clean up
        os.unlink(csv_path)


def test_concat_documents():
    """Test concatenating DocDataFrames"""
    import os
    import sys

    # Add parent directory to path for importing docframe
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from docframe import DocDataFrame, concat_documents

    # Create test DocDataFrames
    df1 = DocDataFrame(
        {"text": ["doc1", "doc2"], "category": ["A", "B"]}, document_column="text"
    )

    df2 = DocDataFrame(
        {"text": ["doc3", "doc4"], "category": ["C", "D"]}, document_column="text"
    )

    # Test vertical concatenation
    result = concat_documents([df1, df2], how="vertical")
    assert len(result) == 4
    assert result.active_document_name == "text"
    assert result.to_polars()["text"].to_list() == ["doc1", "doc2", "doc3", "doc4"]

    # Test with different document column names (should fail)
    df3 = DocDataFrame(
        {"content": ["doc5", "doc6"], "category": ["E", "F"]}, document_column="content"
    )

    with pytest.raises(ValueError, match="same document column name"):
        concat_documents([df1, df3])

    print("✅ concat_documents tests passed")


def test_info_function():
    """Test the info function"""
    import os
    import sys

    # Add parent directory to path for importing docframe
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import docframe

    info_text = docframe.info()
    assert isinstance(info_text, str)
    assert "DocFrame" in info_text
    assert "GeoPandas-inspired" in info_text
    assert "polars" in info_text

    print("✅ info function test passed")


if __name__ == "__main__":
    test_document_column_management()
    test_read_csv_utility()
    test_concat_documents()
    test_info_function()
    print("\n🎉 All utility tests passed!")
