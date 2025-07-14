#!/usr/bin/env python3
"""Simple test runner script for docparse."""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description or ' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"✅ Success: {description or ' '.join(cmd)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description or ' '.join(cmd)} (exit code: {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run docparse tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--build", action="store_true", help="Build Rust extension first")
    parser.add_argument("--install", action="store_true", help="Install test dependencies first")
    
    args = parser.parse_args()
    
    success = True
    
    # Install dependencies if requested
    if args.install:
        success &= run_command(
            ["uv", "pip", "install", "-e", ".[cpu,test]"],
            "Installing test dependencies"
        )
    
    # Build Rust extension if requested
    if args.build:
        success &= run_command(
            ["maturin", "develop"],
            "Building Rust extension"
        )
    
    # Construct pytest command
    pytest_cmd = ["pytest"]
    
    # Add test category markers
    if args.unit:
        pytest_cmd.extend(["-m", "unit"])
    elif args.integration:
        pytest_cmd.extend(["-m", "integration"])
    elif args.e2e:
        pytest_cmd.extend(["-m", "e2e"])
    
    # Add options
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.coverage:
        pytest_cmd.extend([
            "--cov=src/docparse",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=80"
        ])
    
    # Run tests
    description = "Running pytest"
    if args.unit:
        description += " (unit tests only)"
    elif args.integration:
        description += " (integration tests only)"
    elif args.e2e:
        description += " (end-to-end tests only)"
    
    if args.coverage:
        description += " with coverage"
    
    success &= run_command(pytest_cmd, description)
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All operations completed successfully!")
    else:
        print("💥 Some operations failed. Check output above.")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())