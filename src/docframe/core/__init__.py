"""
Core module initialization
"""

# Import text namespace to register it (side effect)
from . import text_namespace  # noqa: F401
from .docframe import DocDataFrame

__all__ = ["DocDataFrame"]
