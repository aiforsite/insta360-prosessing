# Testing Guide

This document describes how to run mock tests before pushing code.

## Quick Start

Run all mock tests:
```bash
python run_mock_tests.py
```

Or run with unittest directly:
```bash
python -m unittest test_processing_runner_mock
```

## Test Files

- **`test_processing_runner_mock.py`** - Mock tests that don't require real API, files, or external dependencies
- **`test_processing_runner.py`** - Integration tests that require real API connections (use with caution)
- **`run_mock_tests.py`** - Convenience script to run mock tests

## What the Mock Tests Cover

The mock tests verify:

1. **Initialization** - VideoProcessor can be initialized with config
2. **API Client Updates** - API credentials can be updated from task data
3. **Full Processing Flow** - Complete task processing pipeline with mocked dependencies
4. **Error Handling** - Proper error handling and cleanup on failures
5. **Reset Tasks** - Reset task handling works correctly
6. **Directory Cleaning** - Local directory cleanup functionality

## Mocked Dependencies

The tests mock:
- **API Clients** - No real API calls are made
- **File Operations** - No real files are created/deleted
- **Video Processing** - MediaSDK operations are mocked
- **Frame Processing** - FFmpeg operations are mocked
- **Route Calculation** - Stella VSLAM operations are mocked
- **External Commands** - All subprocess calls are mocked

## Running Before Push

Before pushing code, always run:
```bash
python run_mock_tests.py
```

This ensures:
- ✅ Code logic is correct
- ✅ No syntax errors
- ✅ Imports work correctly
- ✅ Basic functionality is intact

## Test Structure

Each test:
1. Sets up temporary directories and config
2. Mocks all external dependencies
3. Executes the code under test
4. Verifies expected behavior
5. Cleans up temporary files

## Adding New Tests

To add a new test:

1. Add a new method starting with `test_` in `TestVideoProcessorMock`
2. Use `@patch` decorators to mock dependencies
3. Set up test data and mocks
4. Execute the code
5. Assert expected results

Example:
```python
@patch('processing_runner.APIClient')
def test_new_feature(self, mock_api):
    processor = VideoProcessor(str(self.config_path))
    # ... test code ...
    self.assertTrue(result)
```

## Troubleshooting

If tests fail:
1. Check that all dependencies are mocked
2. Verify test data matches expected format
3. Ensure temporary directories are cleaned up
4. Check for import errors

