#!/usr/bin/env python3


import os
import platform
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils.core import setup
import numpy
source_files = ["src/matching.pyx", "src/helper.c"]


if platform.system() == "Linux":
    exts = Extension(name='matching',
                     sources=source_files,
                     include_dirs=[numpy.get_include()],
                     extra_compile_args=['-fopenmp', "-O3",
                                         ],
                     extra_link_args=['-fopenmp'],
                     language="c",)

elif platform.system() == "Darwin":
    os.environ['CC'] = 'gcc-13'
    exts = Extension(name='matching',
                     sources=source_files,
                     include_dirs=[numpy.get_include()],
                     extra_compile_args=['-fopenmp', "-O3",
                                         ],
                     extra_link_args=['-fopenmp'],
                     language="c",)

setup(name='matching',
      ext_modules=cythonize(exts))
