from setuptools import setup

setup(
    name="causal_graph_comparison",
    version="1.0",
    description="Causal graph comparison for climate models",
    author="Christina Isaicu",
    author_email="c.isaicu@gmail.com",
    packages=["causal_graph_comparison"],  # same as name
    install_requires=[
        "numpy",
        "networkx",
        "matplotlib",
    ],  # external packages as dependencies
)
