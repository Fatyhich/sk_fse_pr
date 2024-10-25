from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='apply_masks',
    ext_modules=cythonize("apply_masks.pyx", compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()]
)