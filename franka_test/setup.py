## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['franka_test',
              'control',
              'control_torch',
              'dist_modules',
              'franka',
              'plotting',
              'vae'],
    package_dir={'':'','control':'scripts/control',
                 'control_torch':'scripts/control_torch',
                 'dist_modules':'scripts/dist_modules',
                 'franka':'scripts/franka',
                 'plotting':'scripts/plotting',
                 'vae':'scripts/vae'})

setup(**setup_args)