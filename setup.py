#!/usr/bin/env python3


import os
import platform
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils.core import setup
from Cython.Distutils import build_ext
import numpy
# from distutils.core import setup, Extension
source_files = ["src/matching.pyx", "src/helper.c"]


# ext_modules = [
#     Extension(
#         "matching",
#         source_files,
#         annotate=True,
#         extra_compile_args=['-fopenmp', "-O3"],
#         extra_link_args=['-fopenmp'],
#         language="c",
#         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
#         extra_info=get_info('npymath')
#     )
# ]

# setup(
#     name='matching',
#     ext_modules=cythonize(ext_modules, language_level="3"),
# )


# setup(
#     cmdclass={'build_ext': build_ext},
#     ext_modules=[
#         Extension("matching",
#                   source_files,
#                   include_dirs=[numpy.get_include()],
#                   extra_compile_args=['-Xpreprocessor', '-fopenmp', "-O3"],
#                   extra_link_args=['-Xpreprocessor', '-fopenmp'],
#                   language="c++",
#                   # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
#                   #  language_leve="3",
#                   )
#     ],
# )


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
