from setuptools import setup, find_packages

setup(name='proteinnetpy',
      version="0.4.0",
      description="A python library for working with ProteinNet data",
      url='https://github.com/allydunham/proteinnetpy',
      author='Alistair Dunham',
      author_email='alistair.dunham@ebi.ac.uk',
      license='Apache 2.0',
      packages=find_packages(),
      install_requires=['numpy', 'pandas', 'biopython'],
      extras_require={'datasets': 'tensorflow>=2.0'},
      entry_points = {
        'console_scripts': ['add_angles_to_proteinnet=proteinnetpy.add_angles:main']
      },
      zip_safe=True)
