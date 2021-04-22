from codecs import open
import os
import sys

from setuptools import setup, find_packages, Extension
from setuptools import dist  # Install numpy right now
dist.Distribution().fetch_build_eggs(['numpy>=1.15.4'])

try:
    import numpy as np
except ImportError:
    exit('Please install numpy>=1.15.4 first.')

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = '0.6.4'

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from README.md
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs]

if sys.platform.startswith("win"):
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized',
                    '-O3', '-ffast-math', '-fopenmp', '-std=c++11']
    link_args = ['-fopenmp', '-std=c++11']

ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [
    Extension('libreco.algorithms._bpr',
              [os.path.join("libreco", "algorithms", "_bpr" + ext)],
              include_dirs=[np.get_include()],
              language="c++",
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('libreco.algorithms._als',
              [os.path.join("libreco", "algorithms", "_als" + ext)],
              include_dirs=[np.get_include()],
              language="c++",
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('libreco.utils._similarities',
              [os.path.join("libreco", "utils", "_similarities" + ext)],
              include_dirs=[np.get_include()],
              language="c++",
              extra_compile_args=compile_args,
              extra_link_args=link_args),
]

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmdclass = {'build_ext': build_ext}
else:
    ext_modules = extensions
    cmdclass = {}

setup(
    name='LibRecommender',
    author='massquantity',
    author_email='jinxin_madie@163.com',
    description=(
        'A collaborative-filtering and content-based recommender system '
        'for both explicit and implicit datasets.'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=__version__,
    url='https://github.com/massquantity/LibRecommender',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Cython',
    ],
    keywords=['Matrix Factorization', 'Collaborative Filtering', 
              'Content-Based', 'Recommender System', 
              'Deep Learning', 'Data Mining'], 

    packages=find_packages(exclude=['test*', 'examples']),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
)

