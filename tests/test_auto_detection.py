"""
Test the automatic document column detection functionality
"""

"""
Test the automatic document column detection functionality
"""

def test_auto_document_detection():
    """Test that the library can automatically detect the document column"""
    import sys
    import os
    # Add parent directory to path for importing docframe
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from docframe import DocDataFrame
    import polars as pl
    
    print("=== Testing Automatic Document Column Detection ===\n")
    
    # Test 1: Clear case - one column has much longer text
    print("1. Clear case - 'content' column has longer text:")
    data1 = {
        'id': [1, 2, 3],
        'title': ['Short title', 'Another title', 'Third title'],
        'content': [
            'This is a much longer piece of content that would clearly be the document text we want to analyze.',
            'Another very long piece of content with lots of words and detailed information about the topic.',
            'Yet another lengthy document with substantial content that should be identified as the main text.'
        ],
        'category': ['news', 'blog', 'article']
    }
    
    df1 = DocDataFrame(data1)  # No document_column specified
    print(f"   Auto-detected document column: '{df1.active_document_name}'")
    print(f"   Sample content: {df1.document[0][:50]}...")
    print()
    
    # Test 2: Manual verification using guess_document_column class method
    print("2. Manual verification using guess_document_column class method:")
    pl_df = pl.DataFrame(data1)
    guessed = DocDataFrame.guess_document_column(pl_df)
    print(f"   Guessed document column: '{guessed}'")
    print()
    
    # Test 3: Only one string column
    print("3. Single string column case:")
    data3 = {
        'id': [1, 2, 3],
        'text': ['Document one content', 'Document two content', 'Document three content'],
        'score': [0.1, 0.2, 0.3]
    }
    
    df3 = DocDataFrame(data3)
    print(f"   Auto-detected document column: '{df3.active_document_name}'")
    print()
    
    # Test 4: No string columns - should fail
    print("4. No string columns case:")
    data4 = {
        'id': [1, 2, 3],
        'score': [0.1, 0.2, 0.3]
    }
    
    try:
        df4 = DocDataFrame(data4)
        print("   ERROR: Should have failed!")
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")
    print()
    
    # Test 5: Manual override still works
    print("5. Manual override test:")
    df5 = DocDataFrame(data1, document_column='title')
    print(f"   Manually set document column: '{df5.active_document_name}'")
    print(f"   Sample content: {df5.document[0]}")
    print()
    
    # Test 6: Show average lengths for verification
    print("6. Show column analysis for verification:")
    lengths_analysis = {}
    pl_df = pl.DataFrame(data1)
    string_columns = [col for col, dtype in pl_df.schema.items() if dtype == pl.Utf8]
    
    for col in string_columns:
        avg_len = pl_df.select(pl.col(col).str.len_chars().mean()).item()
        lengths_analysis[col] = avg_len or 0
    
    print("   Average character lengths in test data:")
    for col, avg_len in sorted(lengths_analysis.items(), key=lambda x: x[1], reverse=True):
        print(f"     {col}: {avg_len:.1f} chars")
    print()
    
    print("✅ Automatic document column detection working!")
    print("\nBenefits:")
    print("  - No need to specify document_column for obvious cases")
    print("  - guess_document_column() helps identify best column")
    print("  - Makes the library more user-friendly")
    
    # Assert key functionality works
    assert df1.active_document_name == 'content', "Should auto-detect 'content' as document column"
    assert guessed == 'content', "guess_document_column should return 'content'"
    assert df3.active_document_name == 'text', "Should auto-detect 'text' as document column"
    assert df5.active_document_name == 'title', "Manual override should work"


if __name__ == "__main__":
    test_auto_document_detection()
