#!/bin/sh

PYTHON_SITE_PATH=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
FILE_NAME="sitecustomize.py"
FULL_PATH="${PYTHON_SITE_PATH}/${FILE_NAME}"

echo "coverage path: ${FULL_PATH}"
cp tests/serving/subprocess_coverage_setup.py "$FULL_PATH"
