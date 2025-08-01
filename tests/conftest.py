"""
Pytest configuration and test setup for DocFrame
"""

import os
import sys

import pytest

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture
def sample_data():
    """Sample data for testing auto-detection"""
    return {
        "id": [1, 2, 3],
        "title": ["Short title", "Another title", "Brief title"],
        "content": [
            "This is a much longer document with detailed content that should be automatically detected as the main document column.",
            "Another very long piece of text with substantial content that represents the primary textual data in this dataset.",
            "A third extensive document containing detailed information and comprehensive coverage of the topic at hand.",
        ],
        "category": ["news", "blog", "article"],
    }


@pytest.fixture
def tweet_data_path():
    """Path to the tweet dataset"""
    return os.path.join(
        project_root, "examples", "data", "ADO", "qldelection2020_candidate_tweets.csv"
    )


@pytest.fixture
def candidate_data_path():
    """Path to the candidate dataset"""
    return os.path.join(project_root, "examples", "data", "ADO", "candidate_info.csv")


# Also provide non-fixture versions for direct import if needed
def get_sample_data():
    """Sample data for testing auto-detection"""
    return {
        "id": [1, 2, 3],
        "title": ["Short title", "Another title", "Brief title"],
        "content": [
            "This is a much longer document with detailed content that should be automatically detected as the main document column.",
            "Another very long piece of text with substantial content that represents the primary textual data in this dataset.",
            "A third extensive document containing detailed information and comprehensive coverage of the topic at hand.",
        ],
        "category": ["news", "blog", "article"],
    }


def get_tweet_data_path():
    """Path to the tweet dataset"""
    return os.path.join(
        project_root, "examples", "data", "ADO", "qldelection2020_candidate_tweets.csv"
    )


def get_candidate_data_path():
    """Path to the candidate dataset"""
    return os.path.join(project_root, "examples", "data", "ADO", "candidate_info.csv")
