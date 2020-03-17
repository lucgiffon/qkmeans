#!/usr/bin/env python
import os
from setuptools import setup, find_packages
import sys
from pathlib import Path

NAME = 'qkmeans'
DESCRIPTION = 'Clustering with learned fast transforms'
LICENSE = 'GNU General Public License v3 (GPLv3)'
# URL = 'https://gitlab.lis-lab.fr/qarma/{}'.format(NAME)
URL = 'https://gitlab.lis-lab.fr/qarma/qalm_qmeans'
AUTHOR = 'Luc Giffon and Valentin Emiya'
AUTHOR_EMAIL = ('valentin.emiya@lis-lab.fr, luc.giffon@lis-lab.fr')
INSTALL_REQUIRES = ['numpy', 'daiquiri', 'matplotlib', 'pandas', 'keras',
                    'docopt', 'pillow', 'scikit-learn==0.22.1', 'psutil', 'yafe', 'python-dotenv', 'click',
                    'xarray', 'tensorflow==1.13.1', 'scipy==1.2.1', 'scikit-luc==2']
# INSTALL_REQUIRES = []
# TODO to be completed
CLASSIFIERS = [
    # 'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Mathematics',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Natural Language :: English',
    'Operating System :: MacOS :: MacOS X ',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6']
PYTHON_REQUIRES = '>=3.5'
EXTRAS_REQUIRE = {
    'dev': ['coverage', 'pytest', 'pytest-cov', 'pytest-randomly'],
    'doc': ['nbsphinx', 'numpydoc', 'sphinx']}
PROJECT_URLS = {'Bug Reports': URL + '/issues',
                'Source': URL}
KEYWORDS = 'clustering, fast transform'  # TODO to be completed

###############################################################################
if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'\n")

if sys.version_info[:2] < (3, 5):
    errmsg = '{} requires Python 3.5 or later ({[0]:d}.{[1]:d} detected).'
    print(errmsg.format(NAME, sys.version_info[:2]))
    sys.exit(-1)


def get_version():
    v_text = open('VERSION').read().strip()
    v_text_formted = '{"' + v_text.replace('\n', '","').replace(':', '":"')
    v_text_formted += '"}'
    v_dict = eval(v_text_formted)
    print(v_text, v_dict)
    return v_dict[NAME]


def set_version(path, VERSION):
    filename = os.path.join(path, '__init__.py')
    buf = ""
    for line in open(filename, "rb"):
        if not line.decode("utf8").startswith("__version__ ="):
            buf += line.decode("utf8")
    f = open(filename, "wb")
    f.write(buf.encode("utf8"))
    f.write(('__version__ = "%s"\n' % VERSION).encode("utf8"))


def setup_package():
    """Setup function"""
    # set version
    VERSION = get_version()

    here = Path(os.path.abspath(os.path.dirname(__file__)))
    with open(here / 'README.rst', encoding='utf-8') as f:
        long_description = f.read()

    mod_dir = Path("code") / NAME
    set_version(mod_dir, get_version())
    setup(name=NAME,
          version=VERSION,
          description=DESCRIPTION,
          long_description=long_description,
          url=URL,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          license=LICENSE,
          classifiers=CLASSIFIERS,
          keywords=KEYWORDS,
          packages=find_packages(where="code", exclude=['doc', 'dev']),
          package_dir={'': "code"},
          install_requires=INSTALL_REQUIRES,
          python_requires=PYTHON_REQUIRES,
          extras_require=EXTRAS_REQUIRE,
          project_urls=PROJECT_URLS)


if __name__ == "__main__":
    setup_package()
