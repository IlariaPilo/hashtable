#!/bin/bash

# setup script
set -e
cd "$(dirname "$0")"

# build and run tests
./build.sh ht_tests RELEASE
cmake-build-release/src/ht_tests
