from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='humoro',
    version='0.1.0',
    description='',
    long_description=readme,
    author='Philipp Kratzer',
    author_email='philipp.kratzer@ipvs.uni-stuttgart.de',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    package_data={'humoro': ['data/*.yaml', 'data/*.urdf', 'data/*.obj']},
)
