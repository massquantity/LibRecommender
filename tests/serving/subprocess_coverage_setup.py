import os

import coverage

# essential for subprocess parallel coverage like sanic serving
# https://coverage.readthedocs.io/en/latest/subprocess.html
os.environ["COVERAGE_PROCESS_START"] = ".coveragerc"
coverage.process_startup()
