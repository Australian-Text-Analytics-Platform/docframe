"""
Test the new polars text namespace functionality
"""

import polars as pl

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


def test_expr_concordance_basic():
    """Test expression-level concordance returns list-of-struct and explode works."""
    df = pl.DataFrame({
        "text": [
            "The quick brown fox",
            "No animals here",
            "Another fox and a fox",
        ]
    })

    # Non-exploded: list of structs per row
    res = df.select(pl.col("text").text.concordance("fox").alias("conc"))
    conc_series = res.get_column("conc")
    assert conc_series.dtype.base_type() == pl.List
    # lengths per row should be [1, 0, 2]
    lengths = [len(x) if x is not None else 0 for x in conc_series]
    assert lengths == [1, 0, 2]

    # Exploded: manually explode to get one struct per match
    exploded = df.select(
        pl.col("text").text.concordance("fox").list.explode().drop_nulls().alias("conc")
    )
    # Total matches = 3
    assert exploded.height == 3
    assert exploded.schema["conc"].base_type() == pl.Struct


def test_expr_quotation_basic():
    """Test expression-level quotation returns list-of-struct and explode works."""
    df = pl.DataFrame({
        "text": [
            'John said, "Hello."',
            "No quotes here",
            '"Hi there," according to Mary.',
        ]
    })

    res = df.select(pl.col("text").text.quotation().alias("quotes"))
    series = res.get_column("quotes")
    assert series.dtype.base_type() == pl.List
    lengths = [len(x) if x is not None else 0 for x in series]
    # Expect at least 1 quote in row 1 and row 3
    assert lengths[0] >= 1 and lengths[1] == 0 and lengths[2] >= 1

    exploded = df.select(
        pl.col("text").text.quotation().list.explode().drop_nulls().alias("q")
    )
    assert exploded.schema["q"].base_type() == pl.Struct


def test_dataframe_quotation():
    df = pl.DataFrame({
        "text": [
            'Alice wrote, "Great job!"',
            '"Indeed," according to Bob.',
            "Nothing here",
        ],
        "id": [1, 2, 3],
    })

    out = df.text.quotation("text")
    # Schema columns expected
    expected = {
        "speaker",
        "speaker_start_idx",
        "speaker_end_idx",
        "quote",
        "quote_start_idx",
        "quote_end_idx",
        "verb",
        "verb_start_idx",
        "verb_end_idx",
        "quote_type",
    }
    assert expected.issubset(set(out.columns))
    # Should extract at least two quotes from rows 1 and 2
    assert len(out) >= 2


def test_series_concordance_and_quotation_namespace():
    s = pl.Series(
        "text",
        [
            'Alice said, "We leave at dawn." Then Bob murmured, "Fine."',
            "No quotes here but has the word dawn repeated: dawn.",
        ],
    )

    conc = s.text.concordance("dawn", num_left_tokens=2, num_right_tokens=2)
    assert conc.len() == 2
    # Expect a list-of-structs per element
    assert conc.dtype.base_type() == pl.List
    # Explode to verify total matches >= 2
    conc_exploded = (
        pl.DataFrame({"c": conc}).select(pl.col("c").list.explode()).drop_nulls()
    )
    assert conc_exploded.height >= 2

    quot = s.text.quotation()
    assert quot.len() == 2
    assert quot.dtype.base_type() == pl.List
    quot_exploded = (
        pl.DataFrame({"q": quot}).select(pl.col("q").list.explode()).drop_nulls()
    )
    assert quot_exploded.height >= 1


def test_dataframe_namespace():
    """Test that df.text works"""
    df = pl.DataFrame({
        "text": ["Hello World!", "This is a test.", "Another example."],
        "id": [1, 2, 3],
    })

    # Test dataframe namespace
    result = df.text.word_count("text")

    print("DataFrame namespace test:")
    print(result)
    print()


def test_document_shortcut():
    """Test df.document.text works with DocDataFrame"""
    from docframe import DocDataFrame

    df = DocDataFrame({
        "text": ["Hello World!", "This is a test.", "Another example."],
        "id": [1, 2, 3],
    })

    # Test document shortcut with text namespace
    word_counts = df.document.text.word_count()

    print("Document shortcut test:")
    print("Word counts:", word_counts)
    print()


def test_namespace_conversions():
    """Test namespace conversion methods"""
    from docframe import DocDataFrame

    # Test DataFrame.text.to_docdataframe()
    regular_df = pl.DataFrame({
        "article": [
            "The quick brown fox",
            "Jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
        ],
        "author": ["Alice", "Bob", "Charlie"],
        "year": [2020, 2021, 2022],
    })

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
