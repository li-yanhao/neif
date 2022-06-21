#!/usr/bin/env python3
# encoding: utf-8

# from matplotlib.pyplot import annotate
# from setuptools import Extension, setup
# from Cython.Build import cythonize
# from numpy.distutils.misc_util import get_info

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




# def configuration(parent_package='', top_path=None):
#     from numpy.distutils.misc_util import Configuration
#     from numpy.distutils.misc_util import get_info

#     # Necessary for the half-float d-type.
#     info = get_info('npymath')

#     config = Configuration('',
#                            parent_package,
#                            top_path)
#     config.add_extension('matching',
#                           ["src/subpixel_match.c", "src/matching.pyx"],
#                          extra_info=info)

#     return config


# if __name__ == "__main__":
#     from numpy.distutils.core import setup
#     setup(configuration=configuration)



from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("matching",
                 source_files,
                 include_dirs=[numpy.get_include()],
                 define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                 extra_compile_args=['-fopenmp', "-O3"],
                 extra_link_args=['-fopenmp'],
                 language="c"
        )
    ],
)
