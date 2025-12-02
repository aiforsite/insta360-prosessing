#!/usr/bin/env python3
"""
Quick script to run mock tests before pushing.
Usage: python run_mock_tests.py
"""

import sys
import unittest
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import mock tests
from test_processing_runner_mock import TestVideoProcessorMock

def main():
    """Run all mock tests."""
    print("=" * 70)
    print("Running Mock Tests for Processing Runner")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all tests
    suite.addTests(loader.loadTestsFromTestCase(TestVideoProcessorMock))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print("=" * 70)
        return 1

if __name__ == '__main__':
    sys.exit(main())

