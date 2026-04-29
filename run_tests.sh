#!/usr/bin/env bash

set -e

echo "Running all tests in the project..."

pytest -o addopts="" code/tests/ -v
