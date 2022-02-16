#!/usr/bin/env python
"""Script for installing dependencies for PArthENoPE GUI"""

from setuptools import setup

import parthenopegui


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="parthenopegui",
    version=parthenopegui.__version__,
    description="A GUI for PArthENoPE 3.0",
    # long_description_content_type="text/markdown",
    long_description=readme(),
    author=parthenopegui.__author__,
    author_email=parthenopegui.__email__,
    url=parthenopegui.__website__,
    # license="GPL-3.0",
    # keywords="bibliography hep-ph high-energy-physics bibtex",
    packages=["parthenopegui"],
    scripts=["gui3.0.py"],
    package_data={"": ["README.md"], "parthenopegui": ["images/*.png"]},
    install_requires=[
        "appdirs",
        'matplotlib(<3);python_version<"3"',
        'matplotlib;python_version>"3"',
        "numpy",
        "pyside2(>=5.14.0)",
        "shiboken2(>=5.14.0)",
        "six",
        'mock;python_version<"3"',
        'unittest2;python_version<"3"',
    ],
    provides=["parthenopegui"],
    # data_files=[("parthenopegui",
    # ["LICENSE", "CHANGELOG", 'physbiblio/gui/images/icon.png'])],
)
