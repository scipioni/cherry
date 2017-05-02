import os,sys
from setuptools import setup, find_packages

setup(
    name='cherry',
    version='0.1',
    classifiers=[
        "Programming Language :: Python",
        ],
    author='Stefano Scipioni',
    author_email='devnull@csgalileo.org',
    url='http://www.csgalileo.org',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[],
    #entry_points={
    #    'console_scripts':
    #        ['giano-ocr = gianoocr.process:run',
    #         'giano-ocr-train = gianoocr.process:train_knn'],
    #    },
)

