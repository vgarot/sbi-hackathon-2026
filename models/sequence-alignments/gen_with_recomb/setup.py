
from setuptools import setup,Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        "generation",
        ["generation.pyx"],
        language="c++",  
        extra_compile_args=['-fopenmp'], 
        extra_link_args=['-fopenmp']
    )
]

setup(ext_modules=cythonize(extensions),
      install_requires=[
            "biopython",
            "torch",
            "random",
            "numpy",
            "math"
      ])