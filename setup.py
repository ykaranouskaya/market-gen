from setuptools import setup

VERSION = '0.0.1'

NAME = 'market_gen'

setup(
    name=NAME,
    version=VERSION,
    description='Transformer based model for market data generation',
    url='',
    author='Yuliya Karanouskaya',
    author_email='y.karanouskaya@gmail.com',
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow>=2.0b",
        "matplotlib"
    ],
    include_package_data=True,
)