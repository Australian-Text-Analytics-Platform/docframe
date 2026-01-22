# DocFrame Testing Guide

**Scope statement:** This page explains how to run and extend the DocFrame test suite.

## 1) Test structure

**Question:** *How is the test suite organized?*

**Answer:** Tests are grouped by feature area (core classes, namespaces, I/O, and utilities) under `docframe/tests/`.

## 2) Running tests

**Question:** *What is the standard way to run tests?*

**Answer:** Use the project’s preferred test runner or `pytest` from the DocFrame directory.

## 3) Adding tests

**Question:** *Where should new tests go?*

**Answer:** Add a new `test_*.py` file in `docframe/tests/` and focus on one feature per file.

## Recap

**Question:** *What should I verify before opening a PR?*

**Answer:** Ensure core tests pass and that any new text operation has both eager and lazy coverage.
