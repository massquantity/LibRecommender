from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name="cc",
    ext_modules=cythonize([
        Extension("_similarities",
                  ["_similarities.pyx"],
                  include_dirs=[np.get_include()],
                  language="c++",
                  extra_compile_args=["-fopenmp", "-std=c++11", '-O3'],
                  extra_link_args=["-fopenmp", "-std=c++11"]),
    ], annotate=True)
)

# cythonize("cc.pyx", annotate=True),

