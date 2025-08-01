"""Tests for text_utils module"""

import pytest
import polars as pl
import numpy as np

import docframe as dp


class TestComputeTokenFrequencies:
    """Test the compute_token_frequencies function"""

    def test_basic_functionality(self):
        """Test basic token frequency computation"""
        df1 = dp.DocDataFrame({"text": ["hello world", "hello there"]})
        df2 = dp.DocDataFrame({"text": ["world peace", "hello world"]})
        
        frames = {"frame1": df1, "frame2": df2}
        result = dp.compute_token_frequencies(frames)
        
        # Check structure
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "frame1" in result
        assert "frame2" in result
        
        # Check that all frames have the same keys (tokens)
        tokens_frame1 = set(result["frame1"].keys())
        tokens_frame2 = set(result["frame2"].keys())
        assert tokens_frame1 == tokens_frame2
        
        # Check expected tokens
        expected_tokens = {"hello", "world", "there", "peace"}
        assert tokens_frame1 == expected_tokens
        
        # Check frequency counts
        assert result["frame1"]["hello"] == 2
        assert result["frame1"]["world"] == 1
        assert result["frame1"]["there"] == 1
        assert result["frame1"]["peace"] == 0  # Not in frame1
        
        assert result["frame2"]["hello"] == 1
        assert result["frame2"]["world"] == 2
        assert result["frame2"]["there"] == 0  # Not in frame2
        assert result["frame2"]["peace"] == 1

    def test_stop_words_filtering(self):
        """Test stop words filtering functionality"""
        df1 = dp.DocDataFrame({"text": ["the hello world", "the hello there"]})
        df2 = dp.DocDataFrame({"text": ["world peace the", "the hello world"]})
        
        frames = {"frame1": df1, "frame2": df2}
        stop_words = ["the"]
        result = dp.compute_token_frequencies(frames, stop_words=stop_words)
        
        # Check that 'the' is not in the results
        for frame_name in result:
            assert "the" not in result[frame_name]
        
        # Check expected tokens (without 'the')
        expected_tokens = {"hello", "world", "there", "peace"}
        for frame_name in result:
            assert set(result[frame_name].keys()) == expected_tokens

    def test_multiple_stop_words(self):
        """Test filtering multiple stop words"""
        df1 = dp.DocDataFrame({"text": ["the quick brown fox", "a lazy dog"]})
        
        frames = {"frame1": df1}
        stop_words = ["the", "a"]
        result = dp.compute_token_frequencies(frames, stop_words=stop_words)
        
        # Check that stop words are filtered
        assert "the" not in result["frame1"]
        assert "a" not in result["frame1"]
        
        # Check remaining tokens
        expected_tokens = {"quick", "brown", "fox", "lazy", "dog"}
        assert set(result["frame1"].keys()) == expected_tokens

    def test_with_doclazyframe(self):
        """Test with DocLazyFrame objects"""
        # Create DocDataFrames first, then convert to lazy
        df1 = dp.DocDataFrame({"text": ["hello world", "hello there"]})
        df2 = dp.DocDataFrame({"text": ["world peace", "hello world"]})
        lazy_df1 = df1.to_doclazyframe()
        lazy_df2 = df2.to_doclazyframe()
        
        frames = {"lazy1": lazy_df1, "lazy2": lazy_df2}
        result = dp.compute_token_frequencies(frames)
        
        # Check structure
        assert len(result) == 2
        assert "lazy1" in result
        assert "lazy2" in result
        
        # Check tokens consistency
        tokens_lazy1 = set(result["lazy1"].keys())
        tokens_lazy2 = set(result["lazy2"].keys())
        assert tokens_lazy1 == tokens_lazy2

    def test_mixed_frame_types(self):
        """Test with mixed DocDataFrame and DocLazyFrame"""
        df = dp.DocDataFrame({"text": ["hello world", "hello there"]})
        lazy_df = dp.DocDataFrame({"text": ["world peace", "hello world"]}).to_doclazyframe()
        
        frames = {"eager": df, "lazy": lazy_df}
        result = dp.compute_token_frequencies(frames)
        
        # Check structure
        assert len(result) == 2
        assert "eager" in result
        assert "lazy" in result
        
        # Check tokens consistency
        tokens_eager = set(result["eager"].keys())
        tokens_lazy = set(result["lazy"].keys())
        assert tokens_eager == tokens_lazy

    def test_single_frame(self):
        """Test with a single frame"""
        df = dp.DocDataFrame({"text": ["hello world", "hello there"]})
        
        frames = {"single": df}
        result = dp.compute_token_frequencies(frames)
        
        assert len(result) == 1
        assert "single" in result
        assert result["single"]["hello"] == 2
        assert result["single"]["world"] == 1
        assert result["single"]["there"] == 1

    def test_empty_frames_dict(self):
        """Test error handling for empty frames dictionary"""
        with pytest.raises(ValueError, match="At least one frame must be provided"):
            dp.compute_token_frequencies({})

    def test_invalid_frame_type(self):
        """Test error handling for invalid frame types"""
        invalid_frame = pl.DataFrame({"text": ["hello world"]})  # Regular polars DataFrame
        
        frames = {"invalid": invalid_frame}
        with pytest.raises(TypeError, match="must be DocDataFrame or DocLazyFrame"):
            dp.compute_token_frequencies(frames)

    def test_empty_documents(self):
        """Test with empty documents"""
        df1 = dp.DocDataFrame({"text": ["", ""]})
        df2 = dp.DocDataFrame({"text": ["hello world", ""]})
        
        frames = {"empty": df1, "mixed": df2}
        result = dp.compute_token_frequencies(frames)
        
        # Check structure for empty frame
        assert len(result) == 2
        
        # Empty frame should have zero counts for all tokens
        expected_tokens = {"hello", "world"}
        assert set(result["empty"].keys()) == expected_tokens
        assert all(count == 0 for count in result["empty"].values())
        
        # Mixed frame should have proper counts
        assert result["mixed"]["hello"] == 1
        assert result["mixed"]["world"] == 1

    def test_complex_text_content(self):
        """Test with complex text content including punctuation"""
        df1 = dp.DocDataFrame({"text": ["Hello, world!", "It's a beautiful day."]})
        df2 = dp.DocDataFrame({"text": ["World peace is possible.", "Hello everyone!"]})
        
        frames = {"complex1": df1, "complex2": df2}
        result = dp.compute_token_frequencies(frames)
        
        # Check that tokens are properly extracted and lowercased
        assert "hello" in result["complex1"]
        assert "world" in result["complex1"]
        assert "beautiful" in result["complex1"]
        assert "peace" in result["complex1"]
        
        # Verify consistency across frames
        tokens_1 = set(result["complex1"].keys())
        tokens_2 = set(result["complex2"].keys())
        assert tokens_1 == tokens_2

    def test_very_long_token_lists(self):
        """Test performance with longer documents"""
        long_text1 = " ".join(["word"] * 1000 + ["unique1"])
        long_text2 = " ".join(["word"] * 800 + ["unique2"])
        
        df1 = dp.DocDataFrame({"text": [long_text1]})
        df2 = dp.DocDataFrame({"text": [long_text2]})
        
        frames = {"long1": df1, "long2": df2}
        result = dp.compute_token_frequencies(frames)
        
        # Check that large counts are handled correctly
        assert result["long1"]["word"] == 1000
        assert result["long1"]["unique1"] == 1
        assert result["long1"]["unique2"] == 0
        
        assert result["long2"]["word"] == 800
        assert result["long2"]["unique1"] == 0
        assert result["long2"]["unique2"] == 1

    def test_auto_detected_document_column(self):
        """Test with auto-detected document columns"""
        # Create frame with multiple text columns where document column is auto-detected
        df = dp.DocDataFrame({
            "short": ["hi", "bye"],
            "content": ["this is a longer text column", "with more detailed content"],
            "id": [1, 2]
        })
        
        frames = {"auto": df}
        result = dp.compute_token_frequencies(frames)
        
        # Should use the longer text column ('content')
        assert "longer" in result["auto"]
        assert "detailed" in result["auto"]
        assert "hi" not in result["auto"]  # From shorter column

    def test_case_insensitive_stop_words(self):
        """Test that stop words filtering is case insensitive"""
        df = dp.DocDataFrame({"text": ["The Hello WORLD", "the world peace"]})
        
        frames = {"case_test": df}
        stop_words = ["the"]  # lowercase
        result = dp.compute_token_frequencies(frames, stop_words=stop_words)
        
        # Both "The" and "the" should be filtered
        assert "the" not in result["case_test"]
        assert "The" not in result["case_test"]
        
        # Other tokens should remain
        assert "hello" in result["case_test"]
        assert "world" in result["case_test"]
        assert "peace" in result["case_test"]
