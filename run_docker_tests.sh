#!/usr/bin/env bash

set -e

echo "Building Docker image for testing (this will install all dependencies in a clean container)..."
docker build -t dora-repro-test -f Dockerfile.test .

echo "----------------------------------------"
echo "Running tests inside Docker..."
echo "----------------------------------------"
docker run --rm dora-repro-test
