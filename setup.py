from codecs import open
import glob
import logging
import os
import platform
import sys

import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


NAME = "LibRecommender"
VERSION = "0.10.2"


here = os.path.abspath(os.path.dirname(__file__))


# Get the long description from README.md
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# get the dependencies and installs
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")


install_requires = [x.strip() for x in all_reqs]


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
    name=NAME,
    author="massquantity",
    author_email="jinxin_madie@163.com",
    description=(
        "A collaborative-filtering and content-based recommender system "
        "for both explicit and implicit datasets."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=VERSION,
    url="https://github.com/massquantity/LibRecommender",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Cython",
    ],
    keywords=[
        "Matrix Factorization",
        "Collaborative Filtering",
        "Content-Based",
        "Recommender System",
        "Deep Learning",
        "Data Mining",
    ],
    packages=find_packages(exclude=["test*", "examples"]),
    setup_requires=["Cython>=0.29"],
    include_package_data=True,
    ext_modules=cythonize(extensions),
    cmdclass={"build_ext": build_ext},
    install_requires=install_requires,
)
