from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="spinopt",
    version="24.5.0",
    description="Scipy interface to NLOPT",
    url="https://github.com/mvds314/spinopt",
    author="Martin van der Schans",
    license="BSD",
    keywords="Optimization",
    packages=["spinopt"],
    install_requires=["nlopt", "numpy"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
