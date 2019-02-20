from distutils.core import setup

import sys
if sys.version_info < (2,7):
  sys.exit('Sorry, Python < 2.7 is not supported')

setup(
  name        = 'molgrid',
  version     = '${PACKAGE_VERSION}', # TODO: might want to use commit ID here
  author='David Ryan Koes and Jocelyn Sunseri',
  author_email='dkoes@pitt.edu',
  packages    = ['molgrid'],
  package_dir = {
    '': '${CMAKE_CURRENT_BINARY_DIR}'
  },
  package_data = {
    '': ['molgrid.so',]
  }
)
