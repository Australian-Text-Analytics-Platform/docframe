# DocFrame Test Suite

This directory contains the test suite for the DocFrame library.

## Test Structure

- **test_core.py** - Core functionality tests for DocSeries and DocDataFrame classes
- **test_namespace.py** - Tests for polars namespace extensions (`.text` accessor)
- **test_auto_detection.py** - Tests for automatic document column detection
- **test_io.py** - Tests for I/O operations (read/write CSV, Parquet, JSON, etc.)
- **test_comprehensive.py** - Comprehensive integration tests
- **test_library.py** - Basic library functionality tests
- **test_utilities.py** - Tests for utility functions
- **conftest.py** - Shared pytest fixtures and test data

## Running Tests

### Using the test runner (recommended):
```bash
# Run all tests with pytest (if available)
python run_tests.py

# Run tests manually without pytest
python run_tests.py --manual

# Run with coverage report
python run_tests.py --coverage

# Run specific test file
python run_tests.py -f test_core.py

# Run tests with specific markers
python run_tests.py -m "not slow"
```

### Using pytest directly:
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=docframe --cov-report=term-missing
```

### Running individual test files:
```bash
# Each test file can be run standalone
python tests/test_core.py
python tests/test_namespace.py
```

## Test Markers

Tests can be marked with pytest markers for selective execution:
- `@pytest.mark.slow` - Marks slow-running tests
- `@pytest.mark.integration` - Marks integration tests
- `@pytest.mark.io` - Marks tests that perform I/O operations

## Writing New Tests

1. Create test files with the `test_` prefix
2. Use pytest conventions:
   - Test functions should start with `test_`
   - Test classes should start with `Test`
3. Import the library modules being tested
4. Use fixtures from `conftest.py` for common test data
5. Add appropriate markers for test categorization