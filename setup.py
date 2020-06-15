from setuptools import setup, find_packages

setup(name='proteinnetpy',
      version="0.3.1",
      description="A python library for working with ProteinNet data",
      url='https://github.com/allydunham/proteinnetpy',
      author='Alistair Dunham',
      author_email='alistair.dunham@ebi.ac.uk',
      license='Apache 2.0',
      packages=find_packages(),
      install_requires=['numpy', 'pandas', 'biopython'],
      extras_require={'datasets': 'tensorflow>=2.0'},
      zip_safe=True)
