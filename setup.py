from setuptools import setup, find_packages

setup(name='proteinnetpy',
      version="1.0.0",
      description="A python library for working with ProteinNet data",
      url='https://github.com/allydunham/proteinnetpy',
      author='Alistair Dunham',
      author_email='alistair.dunham@ebi.ac.uk',
      license='Apache 2.0',
      packages=find_packages(),
      install_requires=['numpy', 'biopython'],
      extras_require={'datasets': 'tensorflow>=2.0'},
      entry_points = {
        'console_scripts': ['add_angles_to_proteinnet=proteinnetpy.scripts.add_angles:main',
                            'proteinnet_to_fasta=proteinnetpy.scripts.to_fasta:main',
                            'filter_proteinnet=proteinnetpy.scripts.filter_proteinnet:main']
      },
      zip_safe=True)
