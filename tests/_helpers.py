"""Helper functions for tests (separated from conftest fixtures).

These are plain utilities that some tests import directly. Keeping them
out of ``conftest.py`` avoids the anti-pattern of importing from
``conftest`` and lets us drop ``__init__.py`` so that ``tests`` is not
treated as a package.
"""

from pathlib import Path


def get_sample_data():
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


def _project_root() -> Path:
    return Path(__file__).parent.parent


def get_tweet_data_path() -> str:
    return str(
        _project_root()
        / "examples"
        / "data"
        / "ADO"
        / "qldelection2020_candidate_tweets.csv"
    )


def get_candidate_data_path() -> str:
    return str(_project_root() / "examples" / "data" / "ADO" / "candidate_info.csv")
