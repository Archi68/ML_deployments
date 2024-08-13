from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "die-casting-classification-model"
DESCRIPTION = "Die Casting classification model package"
URL = "https://veravla-edu.online"
EMAIL = "<info@veravla-edu.online>"
AUTHOR = "<AI GEEKS INTERNS. TEAM #1>"
REQUIRES_PYTHON = ">=3.8.0"

# The rest you shouldn't have to touch too much
# -----------------------------------------------
# Except perhaps the License and Trove Classifiers!
# Trove Classifiers: https://pypi.org/classifiers/
# If you do change the License, remember to change the
# Trove Classifier for that!
long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR = ROOT_DIR / "classification_model"
with open(PACKAGE_DIR / "VERSION") as f:
    __version__ = f.read().strip()
    about["__version__"] = __version__  # Используем __version__ вместо _version


# What package are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"classification_model": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)