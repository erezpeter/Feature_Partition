from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("new_binary_search_perplexity.pyx",force=True), 
)
