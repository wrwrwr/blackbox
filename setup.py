#!/usr/bin/env python3

from setuptools import setup

meta = {
    'name': 'blackbox',
    'version': '0.1.0',
    'packages': ('blackbox',),
    'install_requires': ('cython>=0.24', 'numpy>=1.11', 'scipy>=0.17'),
    'tests_require': ('pytest', 'pytest-benchmark', 'pytest-flake8',
                      'pytest-isort', 'pytest-readme',
                      'flake8-print', 'flake8-todo', 'pep8-naming'),
    'description': "Specialized toolkit for the BlackBox Challenge.",
    'long_description': open('README.md').read(),
    'author': "Wojciech Ruszczewski",
    'author_email': "blackbox@wr.waw.pl",
    'url': "http://github.org/wrwrwr/blackbox",
    'license': "MIT",
    'classifiers': (
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.4"
    )
}

if __name__ == '__main__':
    setup(**meta)
