import setuptools

from mramanpreprocess.constants import VERSION


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setuptools.setup(
    name="mramanpreprocess",
    version=VERSION,
    install_requires=_requires_from_file('requirements.txt'),
    author="Hiroaki Takahashi",
    author_email="aphiloboe@gmail.com",
    description="Package for preprocess Raman image acquired with Phalanx-R",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
