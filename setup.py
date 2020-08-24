from distutils.core import setup

setup(
  name='bcena-loss',
  version='1.0.0',
  packages=['bcena-loss'],
  install_requires=[
    'fastai2',
    'neptune-client',
    'fastscript',
    'psutil',
  ],
)
