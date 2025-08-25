"""Tests for text_utils module"""

import warnings

import polars as pl
import pytest

import docframe as dp

# Suppress external library warnings
warnings.filterwarnings(
    "ignore", message="Importing 'parser.split_arg_string' is deprecated"
)
warnings.filterwarnings(
    "ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'"
)


class TestComputeTokenFrequencies:
    """Test the compute_token_frequencies function"""

    def test_basic_functionality(self):
        df1 = dp.DocDataFrame({"text": ["hello world", "hello there"]})
        df2 = dp.DocDataFrame({"text": ["world peace", "hello world"]})
        frames = {"frame1": df1, "frame2": df2}
        freqs, stats = dp.compute_token_frequencies(frames)
        assert isinstance(freqs, dict) and isinstance(stats, pl.DataFrame)
        assert set(freqs.keys()) == {"frame1", "frame2"}
        tokens_frame1 = set(freqs["frame1"].keys())
        tokens_frame2 = set(freqs["frame2"].keys())
        assert tokens_frame1 == tokens_frame2 == {"hello", "world", "there", "peace"}
        assert freqs["frame1"]["hello"] == 2
        assert freqs["frame2"]["world"] == 2

    def test_stop_words_filtering(self):
        df1 = dp.DocDataFrame({"text": ["the hello world", "the hello there"]})
        df2 = dp.DocDataFrame({"text": ["world peace the", "the hello world"]})
        frames = {"frame1": df1, "frame2": df2}
        freqs, _ = dp.compute_token_frequencies(frames, stop_words=["the"])
        expected_tokens = {"hello", "world", "there", "peace"}
        for name, d in freqs.items():
            assert "the" not in d
            assert set(d.keys()) == expected_tokens

    def test_multiple_stop_words(self):
        df1 = dp.DocDataFrame({"text": ["the quick brown fox", "a lazy dog"]})
        freqs, _ = dp.compute_token_frequencies(
            {"frame1": df1}, stop_words=["the", "a"]
        )
        assert set(freqs["frame1"].keys()) == {"quick", "brown", "fox", "lazy", "dog"}

    def test_with_doclazyframe(self):
        df1 = dp.DocDataFrame({"text": ["hello world", "hello there"]})
        df2 = dp.DocDataFrame({"text": ["world peace", "hello world"]})
        freqs, _ = dp.compute_token_frequencies({
            "lazy1": df1.to_doclazyframe(),
            "lazy2": df2.to_doclazyframe(),
        })
        assert set(freqs.keys()) == {"lazy1", "lazy2"}
        assert set(freqs["lazy1"].keys()) == set(freqs["lazy2"].keys())

    def test_mixed_frame_types(self):
        eager = dp.DocDataFrame({"text": ["hello world", "hello there"]})
        lazy = dp.DocDataFrame({
            "text": ["world peace", "hello world"]
        }).to_doclazyframe()
        freqs, _ = dp.compute_token_frequencies({"eager": eager, "lazy": lazy})
        assert set(freqs["eager"].keys()) == set(freqs["lazy"].keys())

    def test_single_frame(self):
        df = dp.DocDataFrame({"text": ["hello world", "hello there"]})
        freqs, stats = dp.compute_token_frequencies({"single": df})
        assert list(freqs.keys()) == ["single"]
        assert freqs["single"]["hello"] == 2
        assert isinstance(stats, pl.DataFrame)
        # stats should have zero comparison columns but still include tokens
        assert "token" in stats.columns

    def test_empty_frames_dict(self):
        with pytest.raises(ValueError):
            dp.compute_token_frequencies({})

    def test_invalid_frame_type(self):
        with pytest.raises(TypeError):
            dp.compute_token_frequencies({"bad": pl.DataFrame({"text": ["x"]})})

    def test_empty_documents(self):
        df1 = dp.DocDataFrame({"text": ["", ""]})
        df2 = dp.DocDataFrame({"text": ["hello world", ""]})
        freqs, _ = dp.compute_token_frequencies({"empty": df1, "mixed": df2})
        assert all(v == 0 for v in freqs["empty"].values())
        assert freqs["mixed"]["hello"] == 1

    def test_complex_text_content(self):
        df1 = dp.DocDataFrame({"text": ["Hello, world!", "It's a beautiful day."]})
        df2 = dp.DocDataFrame({"text": ["World peace is possible.", "Hello everyone!"]})
        freqs, _ = dp.compute_token_frequencies({"complex1": df1, "complex2": df2})
        for token in ["hello", "world", "beautiful", "peace"]:
            assert token in freqs["complex1"]
        assert set(freqs["complex1"].keys()) == set(freqs["complex2"].keys())

    def test_very_long_token_lists(self):
        long_text1 = " ".join(["word"] * 1000 + ["unique1"])
        long_text2 = " ".join(["word"] * 800 + ["unique2"])
        freqs, _ = dp.compute_token_frequencies({
            "long1": dp.DocDataFrame({"text": [long_text1]}),
            "long2": dp.DocDataFrame({"text": [long_text2]}),
        })
        assert freqs["long1"]["word"] == 1000 and freqs["long2"]["word"] == 800
        assert freqs["long1"]["unique1"] == 1 and freqs["long2"]["unique2"] == 1

    def test_auto_detected_document_column(self):
        df = dp.DocDataFrame({
            "short": ["hi", "bye"],
            "content": [
                "this is a longer text column",
                "with more detailed content",
            ],
            "id": [1, 2],
        })
        freqs, _ = dp.compute_token_frequencies({"auto": df})
        assert (
            "longer" in freqs["auto"]
            and "detailed" in freqs["auto"]
            and "hi" not in freqs["auto"]
        )

    def test_case_insensitive_stop_words(self):
        df = dp.DocDataFrame({"text": ["The Hello WORLD", "the world peace"]})
        freqs, _ = dp.compute_token_frequencies({"case": df}, stop_words=["the"])
        assert (
            "the" not in freqs["case"]
            and "hello" in freqs["case"]
            and "world" in freqs["case"]
            and "peace" in freqs["case"]
        )


class TestTopicVisualization:
    """Tests for topic_visualization utility"""

    def test_basic_two_corpora(self):
        import importlib

        if importlib.util.find_spec("bertopic") is None:
            import pytest

            pytest.skip("BERTopic not installed")
        from docframe.core.text_utils import topic_visualization

        # Use larger corpora to avoid UMAP issues with small datasets
        corpus1 = [
            "transport policy announcement for new rail system",
            "new infrastructure funding for roads and highways",
            "public transport improvements planned across city",
            "government announces major transport investment program",
            "railway infrastructure development project launched",
            "bus network expansion to serve rural areas",
            "metro line construction begins this summer",
            "transport minister outlines five year plan",
            "cycling infrastructure improvements approved",
            "electric vehicle charging stations installed",
            "airport expansion project receives funding",
            "ferry service expansion to outer islands",
        ]
        corpus2 = [
            "health policy update and hospital funding increased",
            "new health infrastructure and medical services launched",
            "public health announcement regarding vaccination program",
            "ministry announces healthcare reform initiative",
            "hospital capacity expansion project approved",
            "mental health services receive additional funding",
            "medical research facility construction begins",
            "healthcare workers receive pay increase",
            "new medical equipment purchased for hospitals",
            "telemedicine services expanded nationwide",
            "pharmaceutical industry regulation updated",
            "healthcare accessibility improved in remote areas",
        ]
        result = topic_visualization([corpus1, corpus2], min_topic_size=2)
        # Basic structure checks
        assert set(result.keys()) == {
            "corpus_sizes",
            "topics",
            "per_corpus_topic_counts",
            "assignments",
            "meta",
        }
        assert result["corpus_sizes"] == [len(corpus1), len(corpus2)]
        # assignments length per corpus
        assert [len(a) for a in result["assignments"]] == [len(corpus1), len(corpus2)]
        # Each topic size list aligns with corpora count
        for t in result["topics"]:
            assert isinstance(t["size"], list) and len(t["size"]) == 2
            assert t["total_size"] == sum(t["size"])
            assert all(isinstance(x, int) and x >= 0 for x in t["size"])
            # Coordinates present
            assert isinstance(t["x"], float) and isinstance(t["y"], float)
        # Ensure at least one topic produced
        assert len(result["topics"]) >= 1

    def test_invalid_input(self):
        import importlib

        if importlib.util.find_spec("bertopic") is None:
            import pytest

            pytest.skip("BERTopic not installed")
        import pytest

        from docframe.core.text_utils import topic_visualization

        with pytest.raises(ValueError):
            topic_visualization([])
        with pytest.raises(ValueError):
            topic_visualization([["doc"], []])  # empty corpus invalid

    def test_custom_min_topic_size(self):
        import importlib

        if importlib.util.find_spec("bertopic") is None:
            import pytest

            pytest.skip("BERTopic not installed")
        from docframe.core.text_utils import topic_visualization

        # Use larger corpora with overlapping vocabulary to help clustering
        corpus1 = [
            "alpha beta gamma topic modeling research",
            "alpha beta advanced text analysis methods",
            "beta gamma machine learning algorithms",
            "alpha gamma natural language processing",
            "beta delta statistical analysis techniques",
            "alpha beta gamma comprehensive study",
            "gamma delta research methodology overview",
            "alpha beta computational linguistics approach",
            "beta gamma advanced data mining",
            "alpha delta text classification methods",
        ]
        corpus2 = [
            "alpha beta delta artificial intelligence systems",
            "beta delta machine learning applications",
            "delta alpha beta neural network architectures",
            "alpha gamma delta deep learning frameworks",
            "beta delta computational intelligence methods",
            "alpha beta delta advanced AI techniques",
            "delta gamma artificial neural networks",
            "alpha beta machine learning optimization",
            "beta delta intelligent system design",
            "delta alpha automated reasoning systems",
        ]
        result = topic_visualization([corpus1, corpus2], min_topic_size=2)
        # All topics should report size lists of length 2
        assert all(len(t["size"]) == 2 for t in result["topics"])
        # Sum of per-corpus topic sizes equals corpus sizes when excluding outlier (-1)
        assigned_counts = [
            sum(t["size"][i] for t in result["topics"]) for i in range(2)
        ]
        # Account for possible outliers: assigned_counts <= corpus_sizes
        for i, total in enumerate(assigned_counts):
            assert total <= result["corpus_sizes"][i]
