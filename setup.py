"""Module installation function.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Parse requirements file. 
# TODO: Consider better alternatives if distributed
with open(here / 'requirements.txt', 'r') as fh:
    required = fh.read().splitlines()

# Get the long description from the README file
with open(here / 'README.md', 'r') as fh:
    long_description = fh.read()

setup(name='mercap',
      version='1.0.0',
      description='Martian Examination for Climate Patterns',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mark Wronkiewicz',
      author_email='',
      python_requires='>=3.9, <4',
      packages=find_packages(),
      include_package_data=True,
      install_requires=required,
      entry_points={
          'console_scripts': [ 
              'parse_mdad = mercap.parse_mdad:cli',
              'extract_climate_measurements = mercap.cli_funcs:extract_climate_measurements',
              ],
          },
      project_urls={'Source': 'https://github.com/JPLMLIA/mercap'}
      )
