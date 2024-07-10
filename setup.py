from setuptools import setup, Extension, find_packages
import pybind11
import os

# Get the directory containing this setup.py
current_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
    Extension(
        'LatticeForge.closest_index_cpp',
        ['LatticeForge/closest_index_cpp.cpp'],
        include_dirs=[
            pybind11.get_include(), 
            pybind11.get_include(user=True),
            os.path.join(current_dir, 'LatticeForge', 'eigen')
        ],
        language='c++',
        extra_compile_args=['-std=c++11', '-Wno-int-in-bool-context', '-Wno-c++11-extensions'],
        extra_link_args=['-std=c++11']
    ),
]

setup(
    name='LatticeForge',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package to create lattice bases, generate lattice point sets, compute Voronoi polytopes, and optimize designs for statistical models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/LatticeForge',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        'numpy',
        'scipy',
        'pybind11'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
