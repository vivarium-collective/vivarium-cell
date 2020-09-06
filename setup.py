import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", 'r') as readme:
    long_description = readme.read()

# to include data in the package, use MANIFEST.in

setup(
    name='vivarium-cell',
    version='0.0.6',
    packages=[
        'cell',
        'cell.bigg_models',
        'cell.compartments',
        'cell.data',
        'cell.data.chromosomes',
        'cell.data.flat',
        'cell.data.flat.media',
        'cell.data.json_files',
        'cell.experiments',
        'cell.library',
        'cell.parameters',
        'cell.plots',
        'cell.processes',
        'cell.reference_data',
        'cell.states'
    ],
    author='Eran Agmon, Ryan Spangler',
    author_email='eagmon@stanford.edu, ryan.spangler@gmail.com',
    url='https://github.com/vivarium-collective/vivarium-cell',
    license='MIT',
    entry_points={
        'console_scripts': []},
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    install_requires=[
        'vivarium-core>=0.0.12',
        'cobra',
        'Arpeggio',
        'parsimonious',
        'stochastic-arrow',
        'pymunk',
        'alphashape'])
