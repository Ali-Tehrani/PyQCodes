from setuptools import setup, Extension, find_packages, Feature


long_description = ""

with open('requirements.txt', 'r') as f_requirements:
    requirements = f_requirements.readlines()
requirements = [r.strip() for r in requirements]

setup(
    name='PyQCodes',
    version=0.1,
    author='',
    author_email='atehrani@uoguelph.ca',
    description=('PyOptiQC - Python Optimized Quantum Codes'
                 'An open source software framework for doing optimized-based quantum '
                 'error-correction.'),
    long_description=long_description,
    install_requires=['setuptools>=18.0', 'cython'] + requirements,
    zip_safe=False,
    license="LICENSE",
    packages=find_packages()
)
