## setup.py

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("bbox_matching.pyx"),
    include_dirs = [numpy.get_include()]
)