import glob
import logging
import os
import platform
import sys

import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


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


setup(
    # packages=find_packages(exclude=["test*", "examples"]),
    ext_modules=cythonize(extensions),
    cmdclass={"build_ext": build_ext},
)
