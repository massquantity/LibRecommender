import glob
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np
import tomli
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup


def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = [
        "/opt/local/bin/g++-mp-[0-9]*.[0-9]*",
        "/opt/local/bin/g++-mp-[0-9]*",
        "/usr/local/bin/g++-[0-9]*.[0-9]*",
        "/usr/local/bin/g++-[0-9]*",
    ]
    if platform.system() == "Darwin":
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return
    else:
        return


if sys.platform.startswith("win"):
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    use_openmp = True
    compile_args = [
        "-Wno-unused-function",
        "-Wno-maybe-uninitialized",
        "-O3",
        "-ffast-math",
    ]
    link_args = []
    if sys.platform.startswith("darwin"):
        gcc = extract_gcc_binaries()
        if gcc is not None:
            logging.info(f"gcc on macOS: {gcc}")
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc
        else:
            use_openmp = False
            logging.warning(
                "No GCC available. Install gcc from Homebrew: brew install gcc."
            )

    if use_openmp:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")
    compile_args.append("-std=c++11")
    link_args.append("-std=c++11")


extensions = [
    Extension(
        "libreco.algorithms._bpr",
        [os.path.join("libreco", "algorithms", "_bpr.pyx")],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        "libreco.algorithms._als",
        [os.path.join("libreco", "algorithms", "_als.pyx")],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        "libreco.utils._similarities",
        [os.path.join("libreco", "utils", "_similarities.pyx")],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

# copy metadata from pyproject.toml
readme = (Path(__file__).parent / "README.md").read_text()
toml_str = (Path(__file__).parent / "pyproject.toml").read_text()
metadata = tomli.loads(toml_str)["project"]

setup(
    name=metadata["name"],
    version=metadata["version"],
    description=metadata["description"],
    author=metadata["authors"][0]["name"],
    author_email=metadata["authors"][0]["email"],
    license=metadata["license"]["text"],
    url=metadata["urls"]["repository"],
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=metadata["classifiers"],
    keywords=metadata["keywords"],
    packages=find_packages(
        where=".",
        include=["libreco*", "libserving*"],
        exclude=["test*", "examples"],
    ),
    include_package_data=True,
    ext_modules=cythonize(extensions),
    cmdclass={"build_ext": build_ext},
)
