from setuptools import setup

setup(
   name='src',
   version='1.0',
   description='Causal graph comparison for climate models',
   author='Christina Isaicu',
   author_email='c.isaicu@gmail.com',
   packages=['src'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
