"""
DocFrame - Document analysis library for LDaCA with polars backend
"""

# Import the namespace modules to register them automatically
from .core import text_namespace  # noqa: F401
from .core.docframe import DocDataFrame, DocLazyFrame
from .core.text_utils import compute_token_frequencies

# Import utilities for convenient access
from .utils import (
    concat_documents,
    from_arrow,
    from_numpy,
    from_pandas,
    info,
    read_avro,
    read_csv,
    read_database,
    read_delta,
    read_excel,
    read_ipc,
    read_json,
    read_ndjson,
    read_parquet,
    scan_csv,
    scan_ndjson,
    scan_parquet,
)

__version__ = "0.1.0"
__all__ = [
    "DocDataFrame",
    "DocLazyFrame",
    "compute_token_frequencies",
    "read_csv",
    "read_parquet",
    "read_json",
    "read_ndjson",
    "read_excel",
    "read_database",
    "read_ipc",
    "read_avro",
    "read_delta",
    "scan_csv",
    "scan_parquet",
    "scan_ndjson",
    "from_pandas",
    "from_arrow",
    "from_numpy",
    "concat_documents",
    "info",
]
