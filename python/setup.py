try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import sys,os,subprocess

def git_pep440_version(path):
    def git_command(args):
        prefix = ['git', '-C', path]
        return subprocess.check_output(prefix + args).decode().strip()
    version_full = git_command(['describe', '--tags'])
    version_tag = git_command(['describe', '--tags', '--abbrev=0'])
    version_tail = version_full[len(version_tag):]
    return version_tag + version_tail.replace('-', '.dev', 1).replace('-', '+', 1)

try:
    version = git_pep440_version(os.path.dirname(os.path.realpath(__file__)))
except subprocess.CalledProcessError:
    sys.exit('Cannot obtain version number from git.')

if sys.version_info < (2,7):
  sys.exit('Sorry, Python < 2.7 is not supported')

setup(
  name        = 'molgrid',
  description='Grid-based molecular modeling library',
  long_description='molgrid can be used to generate several types of tensors '
  'from input molecules, most uniquely three-dimensional voxel grids. Input can '
  'be specified fairly flexibly, with native support for numpy arrays and torch '
  'tensors as well as major molecular file formats via OpenBabel. Output '
  'generation has several options that facilitate obtaining good performance '
  'from machine learning algorithms, including features like data augmentation '
  'and resampling.',
  version     = version, 
  author='David Ryan Koes and Jocelyn Sunseri',
  author_email='dkoes@pitt.edu',
  classifiers=[
      'Development Status :: 4 - Beta',

      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Chemistry',

      'License :: OSI Approved :: Apache Software License',
      'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
  
      'Programming Language :: C++',
      'Programming Language :: Python :: 3',
  ],
  url = 'https://github.com/gnina/libmolgrid',
  project_urls = {'Documentation': 'http://gnina.github.io/libmolgrid'},
  install_requires = [
                 'numpy>=1.16.2',
                 'pytest',
                 'pyquaternion',
                 'importlib-metadata >= 1.0 ; python_version < "3.8"'
      ],
  python_requires = '>=3',
  packages    = ['molgrid'],
  package_dir = {
    '': '${CMAKE_CURRENT_BINARY_DIR}'
  },
  package_data = {
    '': ['molgrid.so',]
  },
  zip_safe = False
)
