import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", 'r') as readme:
    long_description = readme.read()

# to include data in the package, use MANIFEST.in

setup(
    name='vivarium-cell',
    version='0.0.25',
    packages=[
        'vivarium_cell',
        'vivarium_cell.analysis',
        'vivarium_cell.bigg_models',
        'vivarium_cell.composites',
        'vivarium_cell.data',
        'vivarium_cell.data.chromosomes',
        'vivarium_cell.data.flat',
        'vivarium_cell.data.flat.media',
        'vivarium_cell.data.json_files',
        'vivarium_cell.experiments',
        'vivarium_cell.library',
        'vivarium_cell.parameters',
        'vivarium_cell.plots',
        'vivarium_cell.processes',
        'vivarium_cell.reference_data',
        'vivarium_cell.states'
    ],
    author='Eran Agmon, Ryan Spangler',
    author_email='eagmon@stanford.edu, ryan.spangler@gmail.com',
    url='https://github.com/vivarium-collective/vivarium-cell',
    license='MIT',
    entry_points={
        'console_scripts': []},
    description=(
        'A collection of models for simulating cells with Vivarium.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    install_requires=[
        'vivarium-core>=0.0.19',
        'cobra',
        'Arpeggio',
        'parsimonious',
        'stochastic-arrow',
        'pymunk',
        'alphashape'])
