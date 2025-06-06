from setuptools import setup

setup(
    name="cgc",
    version="1.0",
    description="Causal graph comparison for climate models",
    author="Christina Isaicu",
    author_email="c.isaicu@gmail.com",
    packages=["cgc"],  # same as name
    install_requires=[
        "numpy",
        "networkx",
        "matplotlib",
    ],  # external packages as dependencies
)
