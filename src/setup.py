from matplotlib.pyplot import annotate
from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "matching",
        ["matching.pyx"],
        annotate=True,
        extra_compile_args=['-fopenmp', "-O3"],
        extra_link_args=['-fopenmp'],
        language="c++"
    )
]

setup(
    name='matching',
    ext_modules=cythonize(ext_modules),
)

